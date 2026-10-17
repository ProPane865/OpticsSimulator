"""Stack contexts for the native trace kernels and batched tracing.

The stack argument of the trace kernels is an opaque native object
(``optics._native.TraceStack``) built here from per-element contexts; unlike
the old Numba tuple protocol its type does not depend on the element count,
so no padding is needed.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import _native
from .. import constants as _c
from ..numerical._kernels import build_element_context, build_trace_stack
from ..primitive import Ray
from .kernels import _MAX_SEGS_PER_ELEMENT, trace_stack_segments_njit

if TYPE_CHECKING:
    from .element import RefractiveElement


def build_trace_context(
    elements: Sequence["RefractiveElement"],
) -> Any:
    """Flatten an element stack into a native trace stack.

    Returns ``None`` when the stack is empty or any surface lacks the native
    path (or uses an unsupported aperture), so callers can fall back to the
    Python tracer.
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
    return build_trace_stack(ctxs)


def _segments_capacity(n_elements: int) -> int:
    return _MAX_SEGS_PER_ELEMENT * n_elements + 2


def trace_segments_batch(
    elements: Sequence["RefractiveElement"],
    origins: np.ndarray,
    dirs: np.ndarray,
    ray_length: float = 3.0,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None":
    """Parallel per-ray path tracing for rendering.

    Returns ``(starts, ends, kinds, counts)`` with per-ray segment lists, or
    ``None`` when the stack cannot run natively.
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
    _native.trace_segments_batch(
        ctx, origins, dirs, float(ray_length), starts, ends, kinds, counts
    )
    return starts, ends, kinds, counts


def _plane_columns(planes: Sequence[Any]) -> "tuple[np.ndarray, ...]":
    """Stack per-plane geometry into ``(P, 3)`` origins/normals and ``(P,)`` ranges."""
    if not planes:
        empty3 = np.zeros((0, 3), dtype=float)
        empty1 = np.zeros((0,), dtype=float)
        return empty3, empty3.copy(), empty1.copy(), empty1.copy(), empty1.copy(), empty1.copy()
    po = np.ascontiguousarray([p.o for p in planes], dtype=float)
    pn = np.ascontiguousarray([p.n for p in planes], dtype=float)
    u_min = np.ascontiguousarray([p.u_min for p in planes], dtype=float)
    u_max = np.ascontiguousarray([p.u_max for p in planes], dtype=float)
    v_min = np.ascontiguousarray([p.v_min for p in planes], dtype=float)
    v_max = np.ascontiguousarray([p.v_max for p in planes], dtype=float)
    return po, pn, u_min, u_max, v_min, v_max


def measure_planes_raw(
    elements: Sequence["RefractiveElement"],
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
    planes: Sequence[Any],
) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | None":
    """Trace every ray through ``elements`` once and hit every plane.

    Returns ``(us, vs, pws)`` shaped ``(P, N)`` (plane-major), or ``None`` when
    the stack cannot run natively (callers fall back to per-plane tracing).
    """
    ctx = build_trace_context(elements)
    if ctx is None:
        return None
    origins = np.ascontiguousarray(origins, dtype=float)
    dirs = np.ascontiguousarray(dirs, dtype=float)
    powers = np.ascontiguousarray(powers, dtype=float)
    po, pn, u_min, u_max, v_min, v_max = _plane_columns(planes)
    n = origins.shape[0]
    p = len(planes)
    us = np.full((p, n), np.nan)
    vs = np.full((p, n), np.nan)
    pws = np.zeros((p, n))
    _native.measure_planes_multi(
        ctx, origins, dirs, powers, po, pn, u_min, u_max, v_min, v_max, _c.PLANE_EPS, us, vs, pws
    )
    return us, vs, pws


def trace_segments_and_planes(
    elements: Sequence["RefractiveElement"],
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
    ray_length: float,
    planes: Sequence[Any],
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None":
    """Single-trace combined path for rendering and all measurement planes.

    Walks every ray through ``elements`` exactly once and produces both the
    rendering segment buffers ``(starts, ends, kinds, counts)`` and the
    ``(P, N)`` plane hits ``(us, vs, pws)``. Returns ``None`` when the stack
    cannot run natively (callers fall back to the separate trace/measure paths).
    """
    ctx = build_trace_context(elements)
    if ctx is None:
        return None
    origins = np.ascontiguousarray(origins, dtype=float)
    dirs = np.ascontiguousarray(dirs, dtype=float)
    powers = np.ascontiguousarray(powers, dtype=float)
    po, pn, u_min, u_max, v_min, v_max = _plane_columns(planes)
    n = origins.shape[0]
    p = len(planes)
    cap = _segments_capacity(len(elements))
    starts = np.empty((n, cap, 3), dtype=float)
    ends = np.empty((n, cap, 3), dtype=float)
    kinds = np.empty((n, cap), dtype=np.int8)
    counts = np.zeros(n, dtype=np.int64)
    us = np.full((p, n), np.nan)
    vs = np.full((p, n), np.nan)
    pws = np.zeros((p, n))
    _native.trace_segments_and_planes(
        ctx, origins, dirs, powers, float(ray_length), po, pn, u_min, u_max, v_min, v_max,
        _c.PLANE_EPS, starts, ends, kinds, counts, us, vs, pws,
    )
    return starts, ends, kinds, counts, us, vs, pws


_SEGMENT_KIND_NAMES = ("incident", "internal", "exit")


def _trace_stack_path_fast(
    ray: Ray, ctx: Any, ray_length: float, n_elements: int
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
