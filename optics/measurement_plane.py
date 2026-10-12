from collections.abc import Sequence
from typing import Any

import numpy as np

from . import constants as _c
from .ray import Ray
from .refraction import RefractiveElement, TraceResult, internal_rays, traverse_stack
from .vector import normalize

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
        if abs(self.n[2]) <= _EPS or abs(float(np.dot(d, self.n))) <= _EPS:
            return None

        nxnz = (self.n[0] / self.n[2])
        nynz = (self.n[1] / self.n[2])
        A = np.asarray([
            [-1, 0, d[0]],
            [0, -1, d[1]],
            [nxnz, nynz, d[2]],
        ])
        b = np.asarray([-ro[0], -ro[1], nxnz*self.o[0] + nynz*self.o[1] + self.o[2] - ro[2]])
        try:
            u, v, t = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        if not np.isfinite(t) or t <= 0.0:
            return None
        if not (np.isfinite(u) and np.isfinite(v)):
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

        valid = self.contains(u, v) & np.isfinite(power)
        if not np.any(valid):
            return 0.0

        i, j = self.discretize(u[valid], v[valid])
        np.add.at(self.grid, (j, i), power[valid])
        return float(np.sum(power[valid]))

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
        """Trace rays through ``elements`` and accumulate power on the grid."""
        self.reset()
        return self._measure_element_stack(rays, elements)

    def irradiance(self) -> np.ndarray:
        """Power per unit true plane area for each bin."""
        area = self.bin_area / abs(self.n[2])
        return self.grid / area
