"""Flattened stack contexts for the nopython kernels and batched tracing.

The stack argument of the trace kernels is a Numba tuple whose *length* is
part of the static signature: contexts are padded to a fixed capacity with
inert elements so the signature depends only on the capacity, not on the
element count (see ``_CTX_PAD``).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit

from ..numerical._kernels import build_descriptor, build_element_context, build_grid
from ..numerical.parallel import run_in_chunks
from ..primitive import Ray
from .kernels import _MAX_SEGS_PER_ELEMENT, trace_stack_segments_njit

if TYPE_CHECKING:
    from .element import RefractiveElement

_NULL_ELEMENT_CTX: "tuple[Any, ...] | None" = None

# The stack argument of the trace kernels is a Numba tuple whose *length* is
# part of the static signature: without padding, adding one element changes
# the tuple type and forces a full re-JIT of the whole stack kernel chain
# (~10-20 s) on the calling thread. Contexts are therefore padded to a fixed
# capacity with inert elements so the signature depends only on the capacity.
# The kernels iterate the tuple statically, so compile cost grows with the
# capacity; 4 keeps the one-time compile on par with tracing a 4-element
# stack directly, and capacities only change every 4th stack size.
_CTX_PAD = 4


def _padded_capacity(n_elements: int) -> int:
    return max(_CTX_PAD, -(-n_elements // _CTX_PAD) * _CTX_PAD)


def _null_element_context() -> "tuple[Any, ...]":
    """Inert element context used to pad trace stacks to a fixed capacity.

    Its seed grids are empty and its local bounds are inverted, so the
    kernels' hit/containment scans exit immediately and the descriptor's
    cfuncs are never called; they only have to carry the right types.
    """
    global _NULL_ELEMENT_CTX
    if _NULL_ELEMENT_CTX is None:
        from ..numerical.codegen import null_surface_cfuncs

        eval_cfn, deriv_cfn = null_surface_cfuncs()
        desc = build_descriptor(
            eval_cfn,
            deriv_cfn,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            -1.0,
            0.0,
            0.0,
            np.eye(3, 3),
            np.zeros(3),
        )
        grid = build_grid(
            np.zeros((0, 3)),
            np.zeros(0),
            np.zeros(0),
            np.zeros((0, 3)),
            np.zeros(0, dtype=bool),
            np.zeros(0),
            np.zeros(0),
        )
        _NULL_ELEMENT_CTX = build_element_context(
            desc,
            desc,
            grid,
            grid,
            1.0,
            np.ones(3),
            -np.ones(3),
            np.zeros((0, 3, 3)),
            np.zeros((0, 3)),
            True,
            True,
        )
    return _NULL_ELEMENT_CTX


def build_trace_context(
    elements: Sequence["RefractiveElement"],
) -> "tuple[Any, ...] | None":
    """Flatten an element stack into plain data for the nopython kernels.

    The returned tuple is padded with inert contexts up to
    ``_padded_capacity(len(elements))`` so the stack-tuple length stays a
    compile-time constant across element additions (see ``_CTX_PAD``).
    Callers that need the real stack size (e.g. segment buffers) must use
    ``len(elements)``, not ``len`` of the returned tuple.

    Returns ``None`` when the stack is empty or any surface lacks the
    nopython codegen path (or uses an unsupported aperture), so callers can
    fall back to the Python tracer.
    """
    if not elements:
        return None
    ctxs: list[Any] = []
    for el in elements:
        d1 = el.surface1.njit_descriptor
        d2 = el.surface2.njit_descriptor
        if d1 is None or d2 is None:
            return None
        g1 = el.surface1.solver.njit_grid()
        g2 = el.surface2.solver.njit_grid()
        wall = el.sidewall
        if wall is None:
            tri = np.empty((0, 3, 3), dtype=float)
            normals = np.empty((0, 3), dtype=float)
        else:
            tri = wall.triangles()
            normals = wall.face_normals()
        lo, hi = el.local_bounds()
        ctxs.append(
            build_element_context(
                d1,
                d2,
                g1,
                g2,
                float(el.material.refractive_index),
                lo,
                hi,
                tri,
                normals,
                el.surface1.is_graph_over_xy,
                el.surface2.is_graph_over_xy,
            )
        )
    null = _null_element_context()
    target = _padded_capacity(len(ctxs))
    ctxs.extend([null] * (target - len(ctxs)))
    return tuple(ctxs)


def _segments_capacity(n_elements: int) -> int:
    return _MAX_SEGS_PER_ELEMENT * n_elements + 2


@njit(cache=True, nogil=True)
def _trace_stack_paths_chunk(
    stack: Any,
    origins: np.ndarray,
    dirs: np.ndarray,
    ray_length: float,
    starts: np.ndarray,
    ends: np.ndarray,
    kinds: np.ndarray,
    counts: np.ndarray,
    lo: int,
    hi: int,
) -> None:
    for i in range(lo, hi):
        c, ex, fo, fd = trace_stack_segments_njit(
            stack, origins[i], dirs[i], ray_length, starts[i], ends[i], kinds[i]
        )
        counts[i] = c


def trace_segments_batch(
    elements: Sequence["RefractiveElement"],
    origins: np.ndarray,
    dirs: np.ndarray,
    ray_length: float = 3.0,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None":
    """Parallel per-ray path tracing for rendering.

    Returns ``(starts, ends, kinds, counts)`` with per-ray segment lists, or
    ``None`` when the stack cannot run in nopython mode.
    """
    ctx = build_trace_context(elements)
    if ctx is None:
        return None
    origins = np.ascontiguousarray(origins, dtype=float)
    dirs = np.ascontiguousarray(dirs, dtype=float)
    n = origins.shape[0]
    cap = _segments_capacity(len(elements))
    starts = np.empty((n, cap, 3), dtype=float)
    ends = np.empty((n, cap, 3), dtype=float)
    kinds = np.empty((n, cap), dtype=np.int8)
    counts = np.zeros(n, dtype=np.int64)

    def work(lo: int, hi: int) -> None:
        _trace_stack_paths_chunk(
            ctx, origins, dirs, float(ray_length), starts, ends, kinds, counts, lo, hi
        )

    run_in_chunks(work, n)
    return starts, ends, kinds, counts


_SEGMENT_KIND_NAMES = ("incident", "internal", "exit")


def _trace_stack_path_fast(
    ray: Ray, ctx: "tuple[Any, ...]", ray_length: float, n_elements: int
) -> "tuple[list[tuple[np.ndarray, np.ndarray, str]], Ray | None]":
    cap = _segments_capacity(n_elements)
    starts = np.empty((cap, 3), dtype=float)
    ends = np.empty((cap, 3), dtype=float)
    kinds = np.empty(cap, dtype=np.int8)
    count, exited, fo, fd = trace_stack_segments_njit(
        ctx,
        np.asarray(ray.origin, dtype=float),
        np.asarray(ray.direction, dtype=float),
        float(ray_length),
        starts,
        ends,
        kinds,
    )
    segments = [
        (starts[i].copy(), ends[i].copy(), _SEGMENT_KIND_NAMES[int(kinds[i])])
        for i in range(count)
    ]
    if not exited:
        return segments, None
    return segments, Ray(fo, fd, ray.power)
