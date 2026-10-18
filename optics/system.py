"""The optical bench as one piece of persistent native state.

An :class:`OpticalSystem` owns the ordered element stack and the native trace
stack assembled from it. Each element already carries its own native element
(surfaces, sidewall mesh, axis-aligned bounds and graph flags, all built
once), so the system is a thin owning aggregate: it builds the native
``TraceStack`` a single time at construction and every trace or measurement
reuses it. It is the only public tracing entry point: the native stack is an
implementation detail reached through the system's methods, never handed out.

Construction is the validation point. An element whose aperture cannot be
encoded natively raises :class:`ValueError` here, when the system is built,
rather than at the first trace. An empty system is valid and traces straight
rays.
"""

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

from . import _native
from . import constants as _c
from .element import RefractiveElement
from .primitive import Ray

if TYPE_CHECKING:
    from .measurement_plane import MeasurementPlane

_MAX_SEGS_PER_ELEMENT = 2 * _c.MAX_INTERNAL_BOUNCES + 3
_SEGMENT_KIND_NAMES = ("incident", "internal", "exit")


def _segments_capacity(n_elements: int) -> int:
    return _MAX_SEGS_PER_ELEMENT * n_elements + 2


def _plane_columns(planes: "Sequence[MeasurementPlane]") -> "tuple[np.ndarray, ...]":
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


def _trace_segments(
    stack: _native.TraceStack,
    origins: np.ndarray,
    dirs: np.ndarray,
    ray_length: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Parallel per-ray path tracing on the native stack.

    Returns ``(starts, ends, kinds, counts)`` with per-ray segment lists; an
    empty stack produces one straight incident segment per ray.
    """
    origins = np.ascontiguousarray(origins, dtype=float)
    dirs = np.ascontiguousarray(dirs, dtype=float)
    n = origins.shape[0]
    cap = _segments_capacity(len(stack))
    starts = np.empty((n, cap, 3), dtype=float)
    ends = np.empty((n, cap, 3), dtype=float)
    kinds = np.empty((n, cap), dtype=np.int8)
    counts = np.zeros(n, dtype=np.int64)
    _native.trace_segments_batch(
        stack, origins, dirs, float(ray_length), starts, ends, kinds, counts
    )
    return starts, ends, kinds, counts


def _measure_planes(
    stack: _native.TraceStack,
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
    planes: "Sequence[MeasurementPlane]",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Trace every ray through the native stack once and hit every plane.

    Returns ``(us, vs, pws)`` shaped ``(P, N)`` (plane-major); an empty stack
    measures the straight rays directly against each plane.
    """
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
        stack, origins, dirs, powers, po, pn, u_min, u_max, v_min, v_max, _c.PLANE_EPS, us, vs, pws
    )
    return us, vs, pws


def _trace_segments_and_planes(
    stack: _native.TraceStack,
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
    ray_length: float,
    planes: "Sequence[MeasurementPlane]",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Single-trace combined path on the native stack.

    Walks every ray through the stack exactly once and produces both the
    rendering segment buffers ``(starts, ends, kinds, counts)`` and the
    ``(P, N)`` plane hits ``(us, vs, pws)``; an empty stack yields straight
    incident segments and direct plane hits.
    """
    origins = np.ascontiguousarray(origins, dtype=float)
    dirs = np.ascontiguousarray(dirs, dtype=float)
    powers = np.ascontiguousarray(powers, dtype=float)
    po, pn, u_min, u_max, v_min, v_max = _plane_columns(planes)
    n = origins.shape[0]
    p = len(planes)
    cap = _segments_capacity(len(stack))
    starts = np.empty((n, cap, 3), dtype=float)
    ends = np.empty((n, cap, 3), dtype=float)
    kinds = np.empty((n, cap), dtype=np.int8)
    counts = np.zeros(n, dtype=np.int64)
    us = np.full((p, n), np.nan)
    vs = np.full((p, n), np.nan)
    pws = np.zeros((p, n))
    _native.trace_segments_and_planes(
        stack, origins, dirs, powers, float(ray_length), po, pn, u_min, u_max, v_min, v_max,
        _c.PLANE_EPS, starts, ends, kinds, counts, us, vs, pws,
    )
    return starts, ends, kinds, counts, us, vs, pws


def _trace_path(
    ray: Ray, stack: _native.TraceStack, ray_length: float
) -> "tuple[list[tuple[np.ndarray, np.ndarray, str]], Ray | None]":
    """Single-ray ``(segments, final_ray)`` walk on the native stack."""
    cap = _segments_capacity(len(stack))
    starts = np.empty((cap, 3), dtype=float)
    ends = np.empty((cap, 3), dtype=float)
    kinds = np.empty(cap, dtype=np.int8)
    count, exited, fo, fd = _native.trace_stack_segments(
        stack,
        np.ascontiguousarray(ray.origin, dtype=float),
        np.ascontiguousarray(ray.direction, dtype=float),
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


class OpticalSystem:
    """An immutable element stack with a persistent native trace stack."""

    def __init__(self, elements: Sequence[RefractiveElement] = ()) -> None:
        self._elements: tuple[RefractiveElement, ...] = tuple(elements)
        # Building every native element here is what makes configuration
        # errors raise now instead of during the first trace.
        self._stack: _native.TraceStack = _native.build_trace_stack(
            [el.native_element for el in self._elements]
        )

    @property
    def elements(self) -> tuple[RefractiveElement, ...]:
        return self._elements

    def __len__(self) -> int:
        return len(self._elements)

    def __iter__(self) -> Iterator[RefractiveElement]:
        return iter(self._elements)

    # -- tracing -------------------------------------------------------------
    def trace_path(
        self, ray: Ray, ray_length: float = 3.0
    ) -> "tuple[list[tuple[np.ndarray, np.ndarray, str]], Ray | None]":
        """Single-ray ``(segments, final_ray)`` through the stack."""
        return _trace_path(ray, self._stack, ray_length)

    def trace_segments(
        self,
        origins: np.ndarray,
        dirs: np.ndarray,
        ray_length: float = 3.0,
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
        """Batched ``(starts, ends, kinds, counts)`` segment buffers."""
        return _trace_segments(self._stack, origins, dirs, ray_length)

    # -- measurement ---------------------------------------------------------
    def measure_planes(
        self,
        origins: np.ndarray,
        dirs: np.ndarray,
        powers: np.ndarray,
        planes: "Sequence[MeasurementPlane]",
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """One trace hitting every plane: ``(us, vs, pws)`` shaped ``(P, N)``."""
        return _measure_planes(self._stack, origins, dirs, powers, planes)

    def trace_segments_and_planes(
        self,
        origins: np.ndarray,
        dirs: np.ndarray,
        powers: np.ndarray,
        ray_length: float,
        planes: "Sequence[MeasurementPlane]",
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
        """Single-trace combined rendering segments + all-plane hits."""
        return _trace_segments_and_planes(
            self._stack, origins, dirs, powers, ray_length, planes
        )
