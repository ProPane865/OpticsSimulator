from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _native
from . import constants as _c
from .primitive import Ray
from .primitive.vector import normalize

if TYPE_CHECKING:
    from .system import OpticalSystem

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

    def new_result(self) -> "MeasurementResult":
        """A fresh, zeroed detector image accumulated against this plane."""
        return MeasurementResult(self)

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

        u, v, t, ok = _native.plane_intersect(
            np.ascontiguousarray(ro, dtype=float),
            np.ascontiguousarray(d, dtype=float),
            np.ascontiguousarray(self.o, dtype=float),
            np.ascontiguousarray(self.n, dtype=float),
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

    def _project_direct(
        self,
        origins: np.ndarray,
        dirs: np.ndarray,
        powers: np.ndarray,
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Hit parameters ``(u, v)`` and powers for every straight ray.

        Direct (untraced) measurement against this plane; rays travelling
        through an optical system are measured via
        :meth:`~optics.system.OpticalSystem.measure_planes` instead.
        """
        n = origins.shape[0]
        us = np.full(n, np.nan)
        vs = np.full(n, np.nan)
        pws = np.zeros(n)
        _native.measure_direct(
            origins, dirs, powers, self.o, self.n, _EPS, us, vs, pws
        )
        return us, vs, pws

    def _accumulate(
        self,
        result: "MeasurementResult",
        origins: np.ndarray,
        dirs: np.ndarray,
        powers: np.ndarray,
        system: "OpticalSystem | None" = None,
    ) -> None:
        """Trace the ray arrays through ``system`` once and bin into ``result``."""
        if system is None:
            result.accumulate(*self._project_direct(origins, dirs, powers))
            return
        us, vs, pws = system.measure_planes(origins, dirs, powers, [self])
        result.accumulate(us[0], vs[0], pws[0])

    def _into(
        self, result: "MeasurementResult", rays: Sequence[Ray],
        system: "OpticalSystem | None" = None,
    ) -> "MeasurementResult":
        """Trace rays through ``system`` and accumulate power into ``result``."""
        if not rays:
            return result
        origins = np.ascontiguousarray([r.origin for r in rays], dtype=float)
        dirs = np.ascontiguousarray([r.direction for r in rays], dtype=float)
        powers = np.ascontiguousarray([r.power for r in rays], dtype=float)
        self._accumulate(result, origins, dirs, powers, system)
        return result

    def measure(
        self, rays: Sequence[Ray], system: "OpticalSystem | None" = None
    ) -> "MeasurementResult":
        """Trace rays through ``system`` into a fresh detector image."""
        return self._into(self.new_result(), rays, system)

    @staticmethod
    def measure_batch(
        planes: Sequence["MeasurementPlane"],
        rays: Sequence[Ray],
        system: "OpticalSystem | None" = None,
    ) -> "list[MeasurementResult]":
        """Measure every plane, tracing the ray fan through ``system`` once.

        Returns a fresh result per plane in the order of ``planes``. Without a
        system the rays are measured directly against each plane.
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
        return MeasurementPlane.measure_batch_arrays(plane_list, origins, dirs, powers, system)

    @staticmethod
    def measure_batch_arrays(
        planes: Sequence["MeasurementPlane"],
        origins: np.ndarray,
        dirs: np.ndarray,
        powers: np.ndarray,
        system: "OpticalSystem | None" = None,
    ) -> "list[MeasurementResult]":
        """Array-based :meth:`measure_batch` (no per-ray marshalling).

        ``origins``/``dirs`` are ``(N, 3)`` and ``powers`` is ``(N,)``; the fan
        is traced through ``system`` once (or measured directly without one).
        """
        plane_list = list(planes)
        if not plane_list:
            return []
        if origins.shape[0] == 0:
            return [plane.new_result() for plane in plane_list]
        if system is None:
            results = []
            for plane in plane_list:
                result = plane.new_result()
                result.accumulate(*plane._project_direct(origins, dirs, powers))
                results.append(result)
            return results
        us, vs, pws = system.measure_planes(origins, dirs, powers, plane_list)
        results = []
        for q in range(len(plane_list)):
            result = plane_list[q].new_result()
            result.accumulate(us[q], vs[q], pws[q])
            results.append(result)
        return results


class MeasurementResult:
    """Detector image accumulated against one :class:`MeasurementPlane`.

    The accumulated power grid (and its running total) is measurement state:
    each trace produces a fresh result instead of mutating the plane, so a
    plane can be re-measured or edited without touching a displayed result.
    """

    grid: np.ndarray

    def __init__(self, plane: MeasurementPlane) -> None:
        self._plane = plane
        self.grid = np.zeros((plane.rv, plane.ru), dtype=float)
        self._total = 0.0

    @property
    def plane(self) -> MeasurementPlane:
        return self._plane

    @property
    def total(self) -> float:
        """Total power accumulated so far (sum of every :meth:`accumulate`)."""
        return self._total

    def accumulate(self, u: Any, v: Any, power: Any) -> float:
        """Add ray power into bins inside the plane extent; returns the added power."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        v = np.atleast_1d(np.asarray(v, dtype=float))
        power = np.atleast_1d(np.asarray(power, dtype=float))
        p = self._plane
        added = float(
            _native.accumulate(
                self.grid,
                np.ascontiguousarray(u),
                np.ascontiguousarray(v),
                np.ascontiguousarray(power),
                p.u_min,
                p.u_max,
                p.v_min,
                p.v_max,
                p.ru,
                p.rv,
                p.interval_u,
                p.interval_v,
            )
        )
        self._total += added
        return added

    def irradiance(self) -> np.ndarray:
        """Power per unit true plane area for each bin."""
        area = self._plane.bin_area / abs(self._plane.n[2])
        return self.grid / area
