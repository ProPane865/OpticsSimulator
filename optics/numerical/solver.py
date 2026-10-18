"""Ray/surface intersection and parameter inversion drivers.

The kernels run natively (``optics._native``, wrapped by
:mod:`~optics.numerical.intersect` / :mod:`~optics.numerical.invert`); this
module builds the native seed grid (sampled directly from the surface
descriptor) and runs the Python-facing orchestration used by
:class:`~optics.primitive.surface.Surface`.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import constants as _c
from ._kernels import build_seed_grid
from .intersect import surface_intersect_kernel
from .invert import _solve_xy_kernel, parameters_at_kernel, parameters_at_xy_kernel

if TYPE_CHECKING:
    from ..primitive import Surface


class SurfaceSolver:
    surface: "Surface"

    def __init__(self, surface: "Surface") -> None:
        self.surface = surface
        self._native_grid_cache: dict[int, Any] = {}

    def native_grid(self, grid_n: int = _c.SEED_GRID_N) -> Any:
        """Seed grid as a native object for the native kernels.

        The default resolution is the persistent grid the native surface
        samples once at construction. An off-default ``grid_n`` falls back to
        building a dedicated grid from the descriptor (cached per resolution).
        """
        if grid_n == _c.SEED_GRID_N:
            return self.surface.native_grid
        cached = self._native_grid_cache.get(grid_n)
        if cached is None:
            cached = build_seed_grid(self.surface.native_descriptor, grid_n)
            self._native_grid_cache[grid_n] = cached
        return cached

    def parameters_at(self, pos: np.ndarray | Sequence[float]) -> "tuple[float, float] | None":
        """Local parameters whose surface point matches ``pos`` within a tolerance."""
        p = self.surface.transform.point_to_local(np.asarray(pos, dtype=float))
        if not np.all(np.isfinite(p)):
            return None
        u, v, ok = parameters_at_kernel(
            self.surface.native_descriptor, self.native_grid(), np.ascontiguousarray(p, dtype=float)
        )
        if not ok:
            return None
        return float(u), float(v)

    def parameters_at_xy(
        self,
        x: float,
        y: float,
        seed: "tuple[float, float] | None" = None,
    ) -> "tuple[float, float] | None":
        """Local parameters whose point projects to local ``(x, y)``."""
        x = float(x)
        y = float(y)

        if seed is not None:
            result = self._solve_parameters_xy(x, y, float(seed[0]), float(seed[1]))
            if result is not None:
                return result
        desc = self.surface.native_descriptor
        u, v, ok = parameters_at_xy_kernel(desc, self.native_grid(), x, y)
        return (u, v) if ok else None

    def intersect(
        self,
        origin: np.ndarray | Sequence[float],
        direction: np.ndarray | Sequence[float],
    ) -> "tuple[float, float, float] | None":
        """Nearest forward hit of ``origin + t * direction`` as ``(t, u, v)``.

        ``direction`` is normalized internally; building ``Hit``/``Ray``
        objects from the result is left to the caller so this package stays
        independent of :mod:`optics.primitive`.
        """
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
            return None
        d_norm = float(np.linalg.norm(d))
        if d_norm == 0.0:
            return None
        d = d / d_norm
        t, u, v, ok = surface_intersect_kernel(
            self.surface.native_descriptor, self.native_grid(), o, d
        )
        if not ok:
            return None
        return float(t), float(u), float(v)

    def _solve_parameters_xy(
        self, x_target: float, y_target: float, u: float, v: float
    ) -> "tuple[float, float] | None":
        desc = self.surface.native_descriptor
        uu, vv, ok = _solve_xy_kernel(desc, x_target, y_target, u, v)
        if not ok:
            return None
        return uu, vv
