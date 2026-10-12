import numpy as np

from . import constants as _c
from .hit import Hit
from .ray import Ray

_MAX_NEWTON_ITER = _c.MAX_NEWTON_ITER
_STEP_REL_TOL = _c.STEP_REL_TOL
_CLAMP_REL = _c.CLAMP_REL
_CONVERGED_F_NORM = _c.CONVERGED_F_NORM
_DEDUP_UV_TOL = _c.DEDUP_UV_TOL

_CONVERGED = "converged"
_STALLED = "stalled"
_EXHAUSTED = "exhausted"

_PARAM_TOL = _c.PARAM_TOL
_HIT_TOL = _c.HIT_TOL


def _newton(residual, jacobian, x0, clamp=False):
    """Damped Newton iteration for ``residual(x) = 0``.

    Returns ``(x, f_norm, status)`` where ``status`` is one of
    ``_CONVERGED``, ``_STALLED``, or ``_EXHAUSTED``, or
    ``(None, None, None)`` if the iteration failed.
    """
    x = np.array(x0, dtype=float)
    status = _EXHAUSTED
    for _ in range(_MAX_NEWTON_ITER):
        F = residual(x)
        if F is None or not np.all(np.isfinite(F)):
            return None, None, None
        f_norm = float(np.linalg.norm(F))
        if f_norm < _CONVERGED_F_NORM:
            return x, f_norm, _CONVERGED
        J = jacobian(x)
        if J is None or not np.all(np.isfinite(J)):
            return None, None, None
        try:
            if J.shape[0] == J.shape[1]:
                delta = np.linalg.solve(J, -F)
            else:
                delta = np.linalg.lstsq(J, -F, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None, None, None
        if not np.all(np.isfinite(delta)):
            return None, None, None
        step_norm = float(np.linalg.norm(delta))
        if clamp:
            clamp_scale = max(1.0, float(np.max(np.abs(x))))
            if step_norm > _CLAMP_REL * clamp_scale:
                delta = delta * (_CLAMP_REL * clamp_scale / step_norm)
        x = x + delta
        if not np.all(np.isfinite(x)):
            return None, None, None
        if step_norm < _STEP_REL_TOL * max(1.0, float(np.max(np.abs(x)))):
            status = _STALLED
            break
    F = residual(x)
    if F is None or not np.all(np.isfinite(F)):
        return None, None, None
    return x, float(np.linalg.norm(F)), status


class SurfaceSolver:
    def __init__(self, surface):
        self.surface = surface
        self._grid_cache = {}

    def parameters_at(self, pos):
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

    def parameters_at_xy(self, x, y, seed=None):
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

    def intersect(self, ray):
        o = np.asarray(ray.origin, dtype=float)
        d = np.asarray(ray.direction, dtype=float)
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
            return None
        ray = Ray(o, d, ray.power)
        hits = []
        for u0, v0, t0 in self._intersection_seeds(ray):
            hit = self._newton_intersection(ray, u0, v0, t0)
            if hit is None:
                continue
            for i, other in enumerate(hits):
                if abs(hit.u - other.u) < _DEDUP_UV_TOL and abs(hit.v - other.v) < _DEDUP_UV_TOL:
                    if hit.t < other.t:
                        hits[i] = hit
                    break
            else:
                hits.append(hit)
        if not hits:
            return None
        return min(hits, key=lambda h: h.t)

    def _solve_parameters_point(self, p, u, v):
        def residual(x):
            return self.surface.evaluate_local(float(x[0]), float(x[1])) - p

        def jacobian(x):
            ru, rv = self.surface.derivatives_local(float(x[0]), float(x[1]))
            return np.column_stack((ru, rv))

        x, f_norm, status = _newton(residual, jacobian, (u, v))
        if x is None or f_norm >= _PARAM_TOL[status]:
            return None
        u = float(x[0])
        v = float(x[1])
        if not self.surface.accepts_parameters(u, v):
            return None
        return u, v

    def _solve_parameters_xy(self, x_target, y_target, u, v):
        def residual(x):
            p = self.surface.evaluate_local(float(x[0]), float(x[1]))
            return np.array([p[0] - x_target, p[1] - y_target])

        def jacobian(x):
            ru, rv = self.surface.derivatives_local(float(x[0]), float(x[1]))
            return np.array([
                [ru[0], rv[0]],
                [ru[1], rv[1]],
            ])

        x, f_norm, status = _newton(residual, jacobian, (u, v))
        if x is None or f_norm >= _PARAM_TOL[status]:
            return None
        u = float(x[0])
        v = float(x[1])
        if not self.surface.accepts_parameters(u, v):
            return None
        return u, v

    def _aperture_mask(self, uf, vf, pts):
        s = self.surface
        return np.asarray(
            s.aperture.mask(s, uf, vf, pts[:, 0], pts[:, 1], pts[:, 2]),
            dtype=bool,
        )

    def _grid_mask(self, U, V, pts):
        finite = np.all(np.isfinite(pts), axis=1)
        uf = U.ravel()
        vf = V.ravel()
        mask = finite & self.surface.in_domain(uf, vf)
        if self.surface.aperture is not None:
            mask = mask & self._aperture_mask(uf, vf, pts)
        return mask

    def _seed_grid(self, grid_n=_c.SEED_GRID_N):
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

    def _intersection_seeds(self, ray, grid_n=_c.SEED_GRID_N, seed_tol=_c.SEED_TOL, max_seeds=_c.MAX_SEEDS):
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

    def _accept_intersection(self, ray, t, u, v):
        if t > 0.0 and self.surface.accepts_parameters(u, v):
            return Hit(ray.at(t), float(t), float(u), float(v))
        return None

    def _newton_intersection(self, ray, u, v, t):
        def residual(x):
            return self.surface.point(float(x[1]), float(x[2])) - ray.at(float(x[0]))

        def jacobian(x):
            ru, rv = self.surface.tangents(float(x[1]), float(x[2]))
            return np.column_stack((-ray.direction, ru, rv))

        x, f_norm, status = _newton(residual, jacobian, (t, u, v), clamp=True)
        if x is None or f_norm >= _HIT_TOL[status]:
            return None
        return self._accept_intersection(ray, float(x[0]), float(x[1]), float(x[2]))
