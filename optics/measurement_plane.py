from collections.abc import Sequence
from typing import Any

import numpy as np

from . import _native
from . import constants as _c
from .numerical.plane import plane_intersect_kernel
from .numerical.vector import normalize
from .primitive import Ray
from .tracing import RefractiveElement
from .tracing.context import build_trace_context, measure_planes_raw

_EPS = _c.PLANE_EPS


class MeasurementPlane:
    """A planar detector parametrized as a graph over world XY.

    A point on the plane is ``(u, v, z(u, v))`` where ``u``/``v`` are the
    world X/Y coordinates and ``z`` follows the plane equation through
    ``origin`` with ``normal``. This requires a non-zero normal Z component.
    """

    o: np.ndarray
    n: np.ndarray
    ru: int
    rv: int
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    grid: np.ndarray

    def __init__(
        self,
        origin: np.ndarray | Sequence[float],
        normal: np.ndarray | Sequence[float],
        u_range: Sequence[float],
        v_range: Sequence[float],
        resolution: Sequence[int],
    ) -> None:
        self.o = np.asarray(origin, dtype=float)
        if self.o.shape != (3,):
            raise ValueError("Origin must be array of shape (3,)")
        if not np.all(np.isfinite(self.o)):
            raise ValueError("Origin must be finite")

        self.n = normalize(normal)
        if abs(self.n[2]) <= _EPS:
            raise ValueError(
                "Measurement plane normal must have a non-zero Z component"
            )

        self.ru, self.rv = int(resolution[0]), int(resolution[1])
        if self.ru < 1 or self.rv < 1:
            raise ValueError("Resolution must be at least 1 in each direction")

        self.u_min, self.u_max = float(u_range[0]), float(u_range[1])
        self.v_min, self.v_max = float(v_range[0]), float(v_range[1])
        if not (self.u_max > self.u_min and self.v_max > self.v_min):
            raise ValueError("Parameter ranges must be increasing")

        self.grid = np.zeros((self.rv, self.ru), dtype=float)

    @property
    def interval_u(self) -> float:
        return self.u_max - self.u_min

    @property
    def interval_v(self) -> float:
        return self.v_max - self.v_min

    @property
    def bin_area(self) -> float:
        return (self.interval_u / self.ru) * (self.interval_v / self.rv)

    def point(self, u: Any, v: Any) -> np.ndarray:
        """World-space point on the plane for parameters (u, v)."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        nxnz = self.n[0] / self.n[2]
        nynz = self.n[1] / self.n[2]
        z = self.o[2] - nxnz * (u - self.o[0]) - nynz * (v - self.o[1])
        return np.stack(np.broadcast_arrays(u, v, z), axis=-1)

    def intersect(self, ray: Ray) -> "tuple[np.ndarray, float, float] | None":
        d = ray.direction
        ro = ray.origin
        if not (np.all(np.isfinite(ro)) and np.all(np.isfinite(d))):
            return None

        u, v, t, ok = plane_intersect_kernel(
            np.ascontiguousarray(ro, dtype=float),
            np.ascontiguousarray(d, dtype=float),
            np.ascontiguousarray(self.o, dtype=float),
            self.n,
            _EPS,
        )
        if not ok:
            return None

        return ray.at(t), float(u), float(v)

    def contains(self, u: Any, v: Any) -> Any:
        """Boolean mask for parameters inside the plane extent."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return (
            (u >= self.u_min)
            & (u <= self.u_max)
            & (v >= self.v_min)
            & (v <= self.v_max)
        )

    def discretize(self, u: Any, v: Any) -> "tuple[np.ndarray, np.ndarray]":
        """Map parameters to integer bin indices (i, j)."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        i = np.floor((u - self.u_min) / self.interval_u * self.ru)
        j = np.floor((v - self.v_min) / self.interval_v * self.rv)

        i = np.clip(i, 0, self.ru - 1).astype(int)
        j = np.clip(j, 0, self.rv - 1).astype(int)

        return i, j

    def reset(self) -> None:
        self.grid[:] = 0.0

    def accumulate(self, u: Any, v: Any, power: Any) -> float:
        """Add ray power into bins for parameters inside the plane extent."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        v = np.atleast_1d(np.asarray(v, dtype=float))
        power = np.atleast_1d(np.asarray(power, dtype=float))

        return float(
            _native.accumulate(
                self.grid,
                np.ascontiguousarray(u),
                np.ascontiguousarray(v),
                np.ascontiguousarray(power),
                self.u_min,
                self.u_max,
                self.v_min,
                self.v_max,
                self.ru,
                self.rv,
                self.interval_u,
                self.interval_v,
            )
        )

    def _measure_into_arrays(
        self,
        origins: np.ndarray,
        dirs: np.ndarray,
        powers: np.ndarray,
        stack: Any = None,
    ) -> float:
        """Trace ray arrays and accumulate power into the (reset) grid.

        Assumes the grid is already reset. Rays are traced in parallel (native
        threads) through the stack kernel; bin accumulation stays serial so the
        grid is deterministic.
        """
        n = origins.shape[0]
        us = np.full(n, np.nan)
        vs = np.full(n, np.nan)
        pws = np.zeros(n)
        if stack is not None:
            _native.measure_stack(
                stack,
                origins,
                dirs,
                powers,
                self.o,
                self.n,
                _EPS,
                self.u_min,
                self.u_max,
                self.v_min,
                self.v_max,
                us,
                vs,
                pws,
            )
        else:
            _native.measure_direct(
                origins, dirs, powers, self.o, self.n, _EPS, us, vs, pws
            )
        return self.accumulate(us, vs, pws)

    def _measure_into(
        self, rays: Sequence[Ray], elements: Sequence[RefractiveElement] | None = None
    ) -> float:
        """Trace rays through ``elements`` and accumulate power into the grid."""
        if not rays:
            return 0.0
        stack: Any = None
        if elements:
            stack = build_trace_context(elements)
            if stack is None:
                raise ValueError("Element stack cannot be traced natively")
        origins = np.ascontiguousarray([r.origin for r in rays], dtype=float)
        dirs = np.ascontiguousarray([r.direction for r in rays], dtype=float)
        powers = np.ascontiguousarray([r.power for r in rays], dtype=float)
        return self._measure_into_arrays(origins, dirs, powers, stack)

    def measure(
        self, rays: Sequence[Ray], elements: Sequence[RefractiveElement] | None = None
    ) -> float:
        """Trace rays through ``elements`` and accumulate power on the grid."""
        self.reset()
        return self._measure_into(rays, elements)

    @staticmethod
    def measure_batch(
        planes: Sequence["MeasurementPlane"],
        rays: Sequence[Ray],
        elements: Sequence[RefractiveElement] | None = None,
    ) -> "list[float]":
        """Measure every plane, tracing the ray fan through ``elements`` once.

        Each plane's grid is reset then filled; returns the per-plane total in
        the order of ``planes``. Without elements the rays are measured
        directly against each plane.
        """
        plane_list = list(planes)
        if not plane_list:
            return []
        if rays:
            origins = np.ascontiguousarray([r.origin for r in rays], dtype=float)
            dirs = np.ascontiguousarray([r.direction for r in rays], dtype=float)
            powers = np.ascontiguousarray([r.power for r in rays], dtype=float)
        else:
            origins = np.empty((0, 3), dtype=float)
            dirs = np.empty((0, 3), dtype=float)
            powers = np.empty((0,), dtype=float)
        return MeasurementPlane.measure_batch_arrays(plane_list, origins, dirs, powers, elements)

    @staticmethod
    def measure_batch_arrays(
        planes: Sequence["MeasurementPlane"],
        origins: np.ndarray,
        dirs: np.ndarray,
        powers: np.ndarray,
        elements: Sequence[RefractiveElement] | None = None,
    ) -> "list[float]":
        """Array-based :meth:`measure_batch` (no per-ray marshalling).

        ``origins``/``dirs`` are ``(N, 3)`` and ``powers`` is ``(N,)``; the fan
        is traced through ``elements`` once (or measured directly when empty).
        """
        plane_list = list(planes)
        for plane in plane_list:
            plane.reset()
        if not plane_list:
            return []
        if origins.shape[0] == 0:
            return [0.0] * len(plane_list)
        if not elements:
            return [plane._measure_into_arrays(origins, dirs, powers) for plane in plane_list]
        raw = measure_planes_raw(elements, origins, dirs, powers, plane_list)
        if raw is None:
            raise ValueError("Element stack cannot be traced natively")
        us, vs, pws = raw
        return [
            plane.accumulate(us[q], vs[q], pws[q]) for q, plane in enumerate(plane_list)
        ]

    def irradiance(self) -> np.ndarray:
        """Power per unit true plane area for each bin."""
        area = self.bin_area / abs(self.n[2])
        return self.grid / area
