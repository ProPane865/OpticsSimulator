from collections.abc import Sequence
from typing import Any

import numpy as np
from numba import njit, prange

from . import constants as _c
from .plane import _intersect_kernel
from .ray import Ray
from .refraction import (
    RefractiveElement,
    TraceResult,
    _run_in_chunks,
    _trace_stack_to_plane_njit,
    build_trace_context,
    internal_rays,
    traverse_stack,
)
from .vector import normalize

_EPS = _c.PLANE_EPS


@njit(cache=True)
def _accumulate_kernel(
    grid: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    power: np.ndarray,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    ru: int,
    rv: int,
    interval_u: float,
    interval_v: float,
) -> float:
    """Accumulate in-extent ray power into ``grid``; return total added."""
    total = 0.0
    for idx in range(u.shape[0]):
        uu = u[idx]
        vv = v[idx]
        pw = power[idx]
        if not (np.isfinite(uu) and np.isfinite(vv) and np.isfinite(pw)):
            continue
        if uu < u_min or uu > u_max or vv < v_min or vv > v_max:
            continue
        i = int(np.floor((uu - u_min) / interval_u * ru))
        j = int(np.floor((vv - v_min) / interval_v * rv))
        if i < 0:
            i = 0
        elif i > ru - 1:
            i = ru - 1
        if j < 0:
            j = 0
        elif j > rv - 1:
            j = rv - 1
        grid[j, i] += pw
        total += pw
    return total


@njit(parallel=True, cache=True)
def _measure_direct_kernel(
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
    o: np.ndarray,
    n: np.ndarray,
    eps: float,
    us: np.ndarray,
    vs: np.ndarray,
    pws: np.ndarray,
) -> None:
    """Plane-intersect every ray in parallel; write ``(u, v, power)`` per ray."""
    for i in prange(origins.shape[0]):
        u, v, t, ok = _intersect_kernel(origins[i], dirs[i], o, n, eps)
        if ok:
            us[i] = u
            vs[i] = v
            pws[i] = powers[i]


@njit(cache=False, nogil=True)
def _measure_stack_chunk(
    stack: Any,
    origins: np.ndarray,
    dirs: np.ndarray,
    powers: np.ndarray,
    o: np.ndarray,
    n: np.ndarray,
    eps: float,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    us: np.ndarray,
    vs: np.ndarray,
    pws: np.ndarray,
    lo: int,
    hi: int,
) -> None:
    """Trace rays ``[lo, hi)`` through the stack, then hit the plane."""
    for i in range(lo, hi):
        out_o = np.empty(3)
        out_d = np.empty(3)
        found = _trace_stack_to_plane_njit(
            stack,
            origins[i],
            dirs[i],
            o,
            n,
            eps,
            u_min,
            u_max,
            v_min,
            v_max,
            out_o,
            out_d,
        )
        if not found:
            continue
        u, v, t, ok = _intersect_kernel(out_o, out_d, o, n, eps)
        if ok:
            us[i] = u
            vs[i] = v
            pws[i] = powers[i]


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

        u, v, t, ok = _intersect_kernel(
            np.asarray(ro, dtype=float), np.asarray(d, dtype=float), self.o, self.n, _EPS
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
            _accumulate_kernel(
                self.grid,
                u,
                v,
                power,
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

        Rays are traced in parallel (``prange``) through the nopython stack
        kernel when every surface supports it; bin accumulation stays serial
        so the grid is deterministic.
        """
        self.reset()
        if not rays:
            return 0.0
        stack: tuple[Any, ...] | None = ()
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
        if stack:

            def work(lo: int, hi: int) -> None:
                _measure_stack_chunk(
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
                    lo,
                    hi,
                )

            _run_in_chunks(work, len(rays))
        else:
            _measure_direct_kernel(
                origins, dirs, powers, self.o, self.n, _EPS, us, vs, pws
            )
        return self.accumulate(us, vs, pws)

    def irradiance(self) -> np.ndarray:
        """Power per unit true plane area for each bin."""
        area = self.bin_area / abs(self.n[2])
        return self.grid / area
