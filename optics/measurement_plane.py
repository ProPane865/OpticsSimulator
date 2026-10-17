from collections.abc import Sequence
from typing import Any

import numpy as np

from . import _native
from . import constants as _c
from .numerical.plane import plane_intersect_kernel
from .numerical.vector import normalize
from .primitive import Ray
from .tracing import (
    RefractiveElement,
    TraceResult,
    internal_rays,
    traverse_stack,
)
from .tracing.context import build_trace_context

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

    def _entry_limit(self, ray: Ray) -> float:
        hit = self.intersect(ray)
        if hit is None:
            return -1.0
        return float(np.dot(hit[0] - ray.origin, ray.direction))

    def _internal_ray_at_plane(self, result: TraceResult) -> Ray | None:
        """The ray traveling inside ``result``'s element that reaches the plane.

        Returns ``None`` when the plane is not crossed by the internal path,
        i.e. the ray leaves the element before it gets there.
        """
        for segment, endpoint in internal_rays(result):
            hit = self.intersect(segment)
            if hit is None:
                continue
            point, u, v = hit
            if not bool(self.contains(u, v)):
                continue
            t = float(np.dot(point - segment.origin, segment.direction))
            if t <= 0.0:
                continue
            if endpoint is not None and t > float(np.linalg.norm(endpoint - segment.origin)):
                continue
            return segment
        return None

    def _trace_to_plane(
        self, ray: Ray, elements: Sequence[RefractiveElement]
    ) -> Ray | None:
        """Trace a ray through elements in geometric order up to the plane.

        Only elements entered before the plane along the current ray are
        traced, so a plane placed between elements sees the beam at its
        own position regardless of the element list order. When the plane
        cuts through an element, the internal ray crossing the plane is
        returned instead of the ray leaving that element.
        """
        current: Ray | None = ray
        for _, result in traverse_stack(ray, elements, self._entry_limit):
            internal = self._internal_ray_at_plane(result)
            if internal is not None:
                return internal
            current = result.outgoing_ray
        return current

    def _measure_element_stack(
        self, rays: Sequence[Ray], elements: Sequence[RefractiveElement] | None = None
    ) -> float:
        total = 0.0
        for ray in rays:
            if elements:
                traced = self._trace_to_plane(ray, elements)
                if traced is None:
                    continue
            else:
                traced = ray
            hit = self.intersect(traced)
            if hit is None:
                continue
            _, u, v = hit
            total += self.accumulate(u, v, traced.power)

        return total

    def measure(
        self, rays: Sequence[Ray], elements: Sequence[RefractiveElement] | None = None
    ) -> float:
        """Trace rays through ``elements`` and accumulate power on the grid.

        Rays are traced in parallel (native threads) through the stack kernel
        when every surface supports it; bin accumulation stays serial so the
        grid is deterministic.
        """
        self.reset()
        if not rays:
            return 0.0
        stack: Any = None
        if elements:
            stack = build_trace_context(elements)
            if stack is None:
                return self._measure_element_stack(rays, elements)
        origins = np.ascontiguousarray([r.origin for r in rays], dtype=float)
        dirs = np.ascontiguousarray([r.direction for r in rays], dtype=float)
        powers = np.ascontiguousarray([r.power for r in rays], dtype=float)
        us = np.full(len(rays), np.nan)
        vs = np.full(len(rays), np.nan)
        pws = np.zeros(len(rays))
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

    def irradiance(self) -> np.ndarray:
        """Power per unit true plane area for each bin."""
        area = self.bin_area / abs(self.n[2])
        return self.grid / area
