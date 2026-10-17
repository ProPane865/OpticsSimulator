"""Ray/surface intersection and parameter inversion drivers.

The kernels run natively (``optics._native``, wrapped by
:mod:`~optics.numerical.intersect` / :mod:`~optics.numerical.invert`); this
module builds the seed grids and runs the orchestration: seed selection,
dedup, and the tolerance-checked refinement used by the Python-facing API.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from .. import constants as _c
from ._kernels import build_grid
from .intersect import _newton_intersection_njit
from .invert import _solve_xy_njit, parameters_at_xy_njit

if TYPE_CHECKING:
    from ..primitive import Surface


class _SeedGrid(TypedDict):
    Uf: np.ndarray
    Vf: np.ndarray
    pts: np.ndarray
    finite: np.ndarray
    valid: np.ndarray
    Uf_valid: np.ndarray
    Vf_valid: np.ndarray
    P_world: np.ndarray


class SurfaceSolver:
    surface: "Surface"

    def __init__(self, surface: "Surface") -> None:
        self.surface = surface
        self._grid_cache: dict[int, _SeedGrid] = {}
        self._njit_grid_cache: dict[int, Any] = {}

    def njit_grid(self, grid_n: int = _c.SEED_GRID_N) -> Any:
        """Seed grid as a native object for the native kernels."""
        cached = self._njit_grid_cache.get(grid_n)
        if cached is None:
            g = self._seed_grid(grid_n)
            cached = build_grid(
                g["P_world"],
                g["Uf_valid"],
                g["Vf_valid"],
                g["pts"],
                g["finite"],
                g["Uf"],
                g["Vf"],
            )
            self._njit_grid_cache[grid_n] = cached
        return cached

    def parameters_at(self, pos: np.ndarray | Sequence[float]) -> "tuple[float, float] | None":
        """Local parameters whose surface point matches ``pos`` within a tolerance."""
        p = self.surface.transform.point_to_local(np.asarray(pos, dtype=float))
        if not np.all(np.isfinite(p)):
            return None
        u0 = float(p[0])
        v0 = float(p[1])
        if self.surface.accepts_parameters(u0, v0):
            puv = self.surface.evaluate_local(u0, v0)
            if np.all(np.isfinite(puv)) and float(np.linalg.norm(puv - p)) < _c.POINT_MATCH_TOL:
                return u0, v0
        grid = self._seed_grid()
        pts = grid["pts"]
        dist = np.linalg.norm(pts - p, axis=1)
        valid = grid["valid"] & np.isfinite(dist)
        if not valid.any():
            return None
        idx = int(np.argmin(np.where(valid, dist, np.inf)))
        u = float(grid["Uf"][idx])
        v = float(grid["Vf"][idx])
        desc = self.surface.njit_descriptor
        uu, vv, ok = _solve_xy_njit(desc, float(p[0]), float(p[1]), u, v)
        if ok:
            puv = self.surface.evaluate_local(uu, vv)
            if (
                np.all(np.isfinite(puv))
                and float(np.linalg.norm(puv - p)) < _c.POINT_MATCH_TOL
            ):
                return uu, vv
        return None

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
        desc = self.surface.njit_descriptor
        u, v, ok = parameters_at_xy_njit(desc, self.njit_grid(), x, y)
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
        hits: list[tuple[float, float, float]] = []
        for u0, v0, t0 in self._intersection_seeds(o, d):
            hit = self._newton_intersection(o, d, u0, v0, t0)
            if hit is None:
                continue
            t, u, v = hit
            for i, other in enumerate(hits):
                if abs(u - other[1]) < _c.DEDUP_UV_TOL and abs(v - other[2]) < _c.DEDUP_UV_TOL:
                    if t < other[0]:
                        hits[i] = hit
                    break
            else:
                hits.append(hit)
        if not hits:
            return None
        return min(hits, key=lambda h: h[0])

    def _solve_parameters_xy(
        self, x_target: float, y_target: float, u: float, v: float
    ) -> "tuple[float, float] | None":
        desc = self.surface.njit_descriptor
        uu, vv, ok = _solve_xy_njit(desc, x_target, y_target, u, v)
        if not ok:
            return None
        return uu, vv

    def _aperture_mask(self, uf: np.ndarray, vf: np.ndarray, pts: np.ndarray) -> np.ndarray:
        s = self.surface
        assert s.aperture is not None
        return np.asarray(
            s.aperture.mask(s, uf, vf, pts[:, 0], pts[:, 1], pts[:, 2]),
            dtype=bool,
        )

    def _grid_mask(self, U: np.ndarray, V: np.ndarray, pts: np.ndarray) -> np.ndarray:
        finite = np.all(np.isfinite(pts), axis=1)
        uf = U.ravel()
        vf = V.ravel()
        mask = finite & self.surface.in_domain(uf, vf)
        if self.surface.aperture is not None:
            mask = mask & self._aperture_mask(uf, vf, pts)
        return mask

    def _seed_grid(self, grid_n: int = _c.SEED_GRID_N) -> _SeedGrid:
        entry = self._grid_cache.get(grid_n)
        if entry is not None:
            return entry
        s = self.surface
        (u_min, u_max), (v_min, v_max) = s.parameter_range
        us = np.linspace(u_min, u_max, grid_n)
        vs = np.linspace(v_min, v_max, grid_n)
        U, V = np.meshgrid(us, vs)
        Uf = U.ravel()
        Vf = V.ravel()
        pts = s.evaluate_local(U, V).reshape(-1, 3)
        finite = np.all(np.isfinite(pts), axis=1)
        valid = self._grid_mask(U, V, pts)
        entry = {
            "Uf": Uf,
            "Vf": Vf,
            "pts": pts,
            "finite": finite,
            "valid": valid,
            "Uf_valid": Uf[valid],
            "Vf_valid": Vf[valid],
            "P_world": (
                s.transform.points_to_world(pts[valid])
                if valid.any()
                else np.empty((0, 3), dtype=float)
            ),
        }
        self._grid_cache[grid_n] = entry
        return entry

    def _intersection_seeds(
        self,
        o: np.ndarray,
        d: np.ndarray,
        grid_n: int = _c.SEED_GRID_N,
        seed_tol: float = _c.SEED_TOL,
        max_seeds: int = _c.MAX_SEEDS,
    ) -> list[tuple[float, float, float]]:
        grid = self._seed_grid(grid_n)
        if not grid["valid"].any():
            return []
        P = grid["P_world"]
        rel = P - o
        t = rel @ d
        forward = t > 0.0
        if not forward.any():
            return []
        proj = o + t[:, None] * d
        residual = np.linalg.norm(P - proj, axis=1)
        mask = forward & np.isfinite(residual) & (residual <= seed_tol)
        if not mask.any():
            mask = forward & np.isfinite(residual) & (residual <= _c.SEED_TOL_FALLBACK)
            if not mask.any():
                return []
        indices = np.where(mask)[0]
        order = indices[np.argsort(residual[indices])]
        Uf = grid["Uf_valid"]
        Vf = grid["Vf_valid"]
        return [
            (float(Uf[idx]), float(Vf[idx]), float(t[idx]))
            for idx in order[:max_seeds]
        ]

    def _accept_intersection(
        self, t: float, u: float, v: float
    ) -> "tuple[float, float, float] | None":
        if t > 0.0 and self.surface.accepts_parameters(u, v):
            return float(t), float(u), float(v)
        return None

    def _newton_intersection(
        self, o: np.ndarray, d: np.ndarray, u: float, v: float, t: float
    ) -> "tuple[float, float, float] | None":
        desc = self.surface.njit_descriptor
        tt, uu, vv, ok = _newton_intersection_njit(desc, o, d, u, v, t)
        if not ok:
            return None
        return tt, uu, vv
