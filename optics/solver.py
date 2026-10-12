import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from numba import njit

from . import constants as _c
from .hit import Hit
from .ray import Ray

if TYPE_CHECKING:
    from .surface import Surface

_MAX_NEWTON_ITER = _c.MAX_NEWTON_ITER
_STEP_REL_TOL = _c.STEP_REL_TOL
_CLAMP_REL = _c.CLAMP_REL
_CONVERGED_F_NORM = _c.CONVERGED_F_NORM
_DEDUP_UV_TOL = _c.DEDUP_UV_TOL
_MIN_STEP = _c.NEWTON_MIN_STEP
_STALL_STEP = _c.NEWTON_STALL_STEP
_MAX_DAMPED_STEPS = _c.NEWTON_MAX_DAMPED_STEPS

_CONVERGED = "converged"
_STALLED = "stalled"
_EXHAUSTED = "exhausted"

_PARAM_TOL = _c.PARAM_TOL
_HIT_TOL = _c.HIT_TOL

_Residual = Callable[[np.ndarray], "np.ndarray | None"]
_Jacobian = Callable[[np.ndarray], "np.ndarray | None"]


class _SeedGrid(TypedDict):
    Uf: np.ndarray
    Vf: np.ndarray
    pts: np.ndarray
    finite: np.ndarray
    valid: np.ndarray
    Uf_valid: np.ndarray
    Vf_valid: np.ndarray
    P_world: np.ndarray


def _newton_delta(J: np.ndarray, F: np.ndarray) -> np.ndarray | None:
    """Solve ``J delta = -F``, falling back to least squares when rank-deficient."""
    try:
        if J.shape[0] == J.shape[1] and _is_nonsingular(J):
            return np.linalg.solve(J, -F)
        return np.linalg.lstsq(J, -F, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None


@njit(cache=True)
def _is_nonsingular(J: np.ndarray) -> bool:
    """Whether ``det(J)`` is large relative to the product of ``J``'s column norms."""
    rows, cols = J.shape
    if rows != cols:
        return False
    if rows == 3:
        a = J[0, 0]
        b = J[0, 1]
        c = J[0, 2]
        d = J[1, 0]
        e = J[1, 1]
        f = J[1, 2]
        g = J[2, 0]
        h = J[2, 1]
        i = J[2, 2]
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        scale = (
            math.sqrt(a * a + d * d + g * g)
            * math.sqrt(b * b + e * e + h * h)
            * math.sqrt(c * c + f * f + i * i)
        )
    elif rows == 2:
        a = J[0, 0]
        b = J[0, 1]
        c = J[1, 0]
        d = J[1, 1]
        det = a * d - b * c
        scale = math.sqrt(a * a + c * c) * math.sqrt(b * b + d * d)
    else:
        return False
    if scale <= 0.0 or not math.isfinite(scale):
        return False
    return bool(abs(det) / scale > _c.NEWTON_SINGULAR_REL)


def _newton(
    residual: _Residual,
    jacobian: _Jacobian,
    x0: Sequence[float],
    clamp: bool = False,
) -> tuple[np.ndarray, float, str] | None:
    """Damped Newton iteration for ``residual(x) = 0``.

    Steps that leave the surface undefined (outside the domain or aperture)
    are halved until they land on valid parameters, so intersections near a
    patch boundary still converge instead of failing.

    Returns ``(x, f_norm, status)`` where ``status`` is one of
    ``_CONVERGED``, ``_STALLED``, or ``_EXHAUSTED``, or
    ``(None, None, None)`` if the iteration failed.
    """
    x = np.array(x0, dtype=float)
    status = _EXHAUSTED
    damped = 0
    F = residual(x)
    for _ in range(_MAX_NEWTON_ITER):
        if F is None or not np.all(np.isfinite(F)):
            return None
        f_norm = float(np.linalg.norm(F))
        if f_norm < _CONVERGED_F_NORM:
            return x, f_norm, _CONVERGED
        J = jacobian(x)
        if J is None or not np.all(np.isfinite(J)):
            return None
        delta = _newton_delta(J, F)
        if delta is None or not np.all(np.isfinite(delta)):
            return None
        if clamp:
            clamp_scale = max(1.0, float(np.max(np.abs(x))))
            if float(np.linalg.norm(delta)) > _CLAMP_REL * clamp_scale:
                delta = delta * (_CLAMP_REL * clamp_scale / float(np.linalg.norm(delta)))
        step = 1.0
        trial = x + delta
        Ft = residual(trial)
        while Ft is None or not np.all(np.isfinite(Ft)):
            step *= 0.5
            if step < _MIN_STEP:
                return None
            trial = x + step * delta
            Ft = residual(trial)
        if step < 1.0:
            damped += 1
        step_norm = float(np.linalg.norm(step * delta))
        x = trial
        F = Ft
        if step < _STALL_STEP or damped >= _MAX_DAMPED_STEPS:
            # Pinned against the edge of the domain: further iterations crawl.
            status = _STALLED
            break
        if step_norm < _STEP_REL_TOL * max(1.0, float(np.max(np.abs(x)))):
            status = _STALLED
            break
    if F is None or not np.all(np.isfinite(F)):
        return None
    return x, float(np.linalg.norm(F)), status


class SurfaceSolver:
    surface: "Surface"

    def __init__(self, surface: "Surface") -> None:
        self.surface = surface
        self._grid_cache: dict[int, _SeedGrid] = {}

    def parameters_at(self, pos: np.ndarray | Sequence[float]) -> "tuple[float, float] | None":
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
        return self._solve_parameters_point(p, u, v)

    def parameters_at_xy(
        self,
        x: float,
        y: float,
        seed: "tuple[float, float] | None" = None,
    ) -> "tuple[float, float] | None":
        x = float(x)
        y = float(y)

        if seed is not None:
            result = self._solve_parameters_xy(x, y, float(seed[0]), float(seed[1]))
            if result is not None:
                return result

        grid = self._seed_grid()
        pts = grid["pts"]
        finite = grid["finite"]

        if not finite.any():
            return None

        distance2 = np.where(finite, ((pts[:, 0] - x)**2 + (pts[:, 1] - y)**2), np.inf)
        k = min(12, distance2.size)
        order = np.argpartition(distance2, k - 1)[:k]
        order = order[np.argsort(distance2[order])]

        Uf = grid["Uf"]
        Vf = grid["Vf"]

        for idx in order:
            result = self._solve_parameters_xy(x, y, float(Uf[idx]), float(Vf[idx]))

            if result is not None:
                return result

        return None

    def intersect(self, ray: Ray) -> Hit | None:
        o = np.asarray(ray.origin, dtype=float)
        d = np.asarray(ray.direction, dtype=float)
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
            return None
        ray = Ray(o, d, ray.power)
        hits: list[Hit] = []
        for u0, v0, t0 in self._intersection_seeds(ray):
            hit = self._newton_intersection(ray, u0, v0, t0)
            if hit is None:
                continue
            for i, other in enumerate(hits):
                if (
                    abs(hit.u - other.u) < _DEDUP_UV_TOL
                    and abs(hit.v - other.v) < _DEDUP_UV_TOL
                ):
                    if hit.t < other.t:
                        hits[i] = hit
                    break
            else:
                hits.append(hit)
        if not hits:
            return None
        return min(hits, key=lambda h: h.t)

    def _solve_parameters_point(
        self, p: np.ndarray, u: float, v: float
    ) -> "tuple[float, float] | None":
        def residual(x: np.ndarray) -> np.ndarray:
            return self.surface.evaluate_local(float(x[0]), float(x[1])) - p

        def jacobian(x: np.ndarray) -> np.ndarray:
            ru, rv = self.surface.derivatives_local(float(x[0]), float(x[1]))
            return np.column_stack((ru, rv))

        result = _newton(residual, jacobian, (u, v))
        if result is None:
            return None
        x, f_norm, status = result
        if f_norm >= _PARAM_TOL[status]:
            return None
        u = float(x[0])
        v = float(x[1])
        if not self.surface.accepts_parameters(u, v):
            return None
        return u, v

    def _solve_parameters_xy(
        self, x_target: float, y_target: float, u: float, v: float
    ) -> "tuple[float, float] | None":
        def residual(x: np.ndarray) -> np.ndarray:
            p = self.surface.evaluate_local(float(x[0]), float(x[1]))
            return np.array([p[0] - x_target, p[1] - y_target])

        def jacobian(x: np.ndarray) -> np.ndarray:
            ru, rv = self.surface.derivatives_local(float(x[0]), float(x[1]))
            return np.array([
                [ru[0], rv[0]],
                [ru[1], rv[1]],
            ])

        result = _newton(residual, jacobian, (u, v))
        if result is None:
            return None
        x, f_norm, status = result
        if f_norm >= _PARAM_TOL[status]:
            return None
        u = float(x[0])
        v = float(x[1])
        if not self.surface.accepts_parameters(u, v):
            return None
        return u, v

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
        ray: Ray,
        grid_n: int = _c.SEED_GRID_N,
        seed_tol: float = _c.SEED_TOL,
        max_seeds: int = _c.MAX_SEEDS,
    ) -> list[tuple[float, float, float]]:
        grid = self._seed_grid(grid_n)
        if not grid["valid"].any():
            return []
        P = grid["P_world"]
        rel = P - ray.origin
        t = rel @ ray.direction
        forward = t > 0.0
        if not forward.any():
            return []
        proj = ray.at(t[:, None])
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

    def _accept_intersection(self, ray: Ray, t: float, u: float, v: float) -> Hit | None:
        if t > 0.0 and self.surface.accepts_parameters(u, v):
            return Hit(ray.at(t), float(t), float(u), float(v))
        return None

    def _newton_intersection(self, ray: Ray, u: float, v: float, t: float) -> Hit | None:
        def residual(x: np.ndarray) -> np.ndarray:
            return self.surface.point(float(x[1]), float(x[2])) - ray.at(float(x[0]))

        def jacobian(x: np.ndarray) -> np.ndarray:
            ru, rv = self.surface.tangents(float(x[1]), float(x[2]))
            return np.column_stack((-ray.direction, ru, rv))

        result = _newton(residual, jacobian, (t, u, v), clamp=True)
        if result is None:
            return None
        x, f_norm, status = result
        if f_norm >= _HIT_TOL[status]:
            return None
        return self._accept_intersection(ray, float(x[0]), float(x[1]), float(x[2]))
