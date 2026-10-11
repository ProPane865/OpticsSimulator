import numpy as np

from .ray import Ray
from .hit import Hit

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
            if np.all(np.isfinite(puv)) and float(np.linalg.norm(puv - p)) < 1e-8:
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
        for u0, v0, t0 in self._intersection_seeds(ray):
            hit = self._newton_intersection(ray, u0, v0, t0)
            if hit is not None:
                return hit
        return None

    def _solve_parameters_point(self, p, u, v):
        for _ in range(50):
            puv = self.surface.evaluate_local(u, v)
            F = puv - p
            if not np.all(np.isfinite(F)):
                return None
            f_norm = float(np.linalg.norm(F))
            if f_norm < 1e-11:
                if self.surface.accepts_parameters(u, v):
                    return float(u), float(v)
                return None
            ru, rv = self.surface.derivatives_local(u, v)
            if not (np.all(np.isfinite(ru)) and np.all(np.isfinite(rv))):
                return None
            J = np.column_stack((ru, rv))
            try:
                delta = np.linalg.lstsq(J, -F, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(delta)):
                return None
            u += float(delta[0])
            v += float(delta[1])
            if not (np.isfinite(u) and np.isfinite(v)):
                return None
            step_norm = float(np.linalg.norm(delta))
            if step_norm < 1e-14 * max(1.0, abs(u), abs(v)):
                puv = self.surface.evaluate_local(u, v)
                F = puv - p
                if (
                    np.all(np.isfinite(F))
                    and float(np.linalg.norm(F)) < 1e-9
                    and self.surface.accepts_parameters(u, v)
                ):
                    return float(u), float(v)
                return None
        puv = self.surface.evaluate_local(u, v)
        F = puv - p
        if (
            np.all(np.isfinite(F))
            and float(np.linalg.norm(F)) < 1e-8
            and self.surface.accepts_parameters(u, v)
        ):
            return float(u), float(v)
        return None

    def _solve_parameters_xy(self, x_target, y_target, u, v):
        for _ in range(50):
            p = self.surface.evaluate_local(u, v)

            if not np.all(np.isfinite(p)):
                return None

            F = np.array([
                p[0] - x_target,
                p[1] - y_target
            ])

            f_norm = float(np.linalg.norm(F))

            if f_norm < 1e-11:
                if self.surface.accepts_parameters(u, v):
                    return float(u), float(v)
                return None

            ru, rv = self.surface.derivatives_local(u, v)

            if not (
                np.all(np.isfinite(ru))
                and np.all(np.isfinite(rv))
            ):
                return None

            J = np.array([
                [ru[0], rv[0]],
                [ru[1], rv[1]]
            ])

            try:
                delta = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                return None

            if not np.all(np.isfinite(delta)):
                return None

            u += float(delta[0])
            v += float(delta[1])

            if not (np.isfinite(u) and np.isfinite(v)):
                return None

            step_norm = float(np.linalg.norm(delta))

            if step_norm < 1e-14 * max(1.0, abs(u), abs(v)):
                p = self.surface.evaluate_local(u, v)
                F = np.array([
                    p[0] - x_target,
                    p[1] - y_target
                ])
                if (
                    np.all(np.isfinite(F))
                    and float(np.linalg.norm(F)) < 1e-9
                    and self.surface.accepts_parameters(u, v)
                ):
                    return float(u), float(v)
                return None

        p = self.surface.evaluate_local(u, v)

        if not np.all(np.isfinite(p)):
            return None

        F = np.array([
            p[0] - x_target,
            p[1] - y_target
        ])

        if (
            np.all(np.isfinite(F))
            and float(np.linalg.norm(F)) < 1e-8
            and self.surface.accepts_parameters(u, v)
        ):
            return float(u), float(v)

        return None

    def _domain_mask(self, uf, vf):
        s = self.surface
        tu = 1e-9 * max(1.0, abs(s.u_range[1] - s.u_range[0]))
        tv = 1e-9 * max(1.0, abs(s.v_range[1] - s.v_range[0]))
        return (
            (uf >= s.u_range[0] - tu)
            & (uf <= s.u_range[1] + tu)
            & (vf >= s.v_range[0] - tv)
            & (vf <= s.v_range[1] + tv)
        )

    def _aperture_mask(self, uf, vf, pts):
        s = self.surface
        try:
            return np.asarray(
                s.aperture.mask(s, uf, vf, pts[:, 0], pts[:, 1], pts[:, 2]),
                dtype=bool,
            )
        except NotImplementedError:
            return np.array(
                [bool(s.aperture.contains(s, u, v)) for u, v in zip(uf, vf)],
                dtype=bool,
            )

    def _grid_mask(self, U, V, pts):
        finite = np.all(np.isfinite(pts), axis=1)
        uf = U.ravel()
        vf = V.ravel()
        mask = finite & self._domain_mask(uf, vf)
        if self.surface.aperture is not None:
            mask = mask & self._aperture_mask(uf, vf, pts)
        return mask

    def _seed_grid(self, grid_n=64):
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

    def _intersection_seeds(self, ray, grid_n=64, seed_tol=0.75, max_seeds=12):
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
            mask = forward & np.isfinite(residual) & (residual <= 2.0)
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

    def _accept_intersection(self, ray, t, u, v, tol):
        p = self.surface.point(u, v)
        F = p - ray.at(t)
        if (
            np.all(np.isfinite(F))
            and float(np.linalg.norm(F)) < tol
            and t > 0.0
            and self.surface.accepts_parameters(u, v)
        ):
            return Hit(ray.at(t), float(t), float(u), float(v))
        return None

    def _newton_intersection(self, ray, u, v, t):
        for _ in range(50):
            p = self.surface.point(u, v)
            F = p - ray.at(t)
            if not np.all(np.isfinite(F)):
                return None
            if float(np.linalg.norm(F)) < 1e-12:
                return self._accept_intersection(ray, t, u, v, 1e-11)
            ru, rv = self.surface.tangents(u, v)
            if not (np.all(np.isfinite(ru)) and np.all(np.isfinite(rv))):
                return None
            A = np.column_stack((-ray.direction, ru, rv))
            try:
                delta = np.linalg.solve(A, -F)
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(delta)):
                return None
            scale = max(1.0, abs(t), abs(u), abs(v))
            step_norm = float(np.linalg.norm(delta))
            if step_norm > 0.5 * scale:
                delta = delta * (0.5 * scale / step_norm)
            t += float(delta[0])
            u += float(delta[1])
            v += float(delta[2])
            if not (np.isfinite(t) and np.isfinite(u) and np.isfinite(v)):
                return None
            if step_norm < 1e-14 * scale:
                return self._accept_intersection(ray, t, u, v, 1e-10)
        return self._accept_intersection(ray, t, u, v, 1e-10)
