import numpy as np

from .ray import Ray
from .hit import Hit

class SurfaceSolver:
    def __init__(self, surface):
        self.surface = surface

    def parameters_at(self, pos):
        p = self.surface.transform.point_to_local(np.asarray(pos, dtype=float))
        if not np.all(np.isfinite(p)):
            return None
        u0 = float(p[0])
        v0 = float(p[1])
        if self.surface.contains_point(u0, v0):
            puv = self.surface.evaluate_local(u0, v0)
            if np.all(np.isfinite(puv)) and float(np.linalg.norm(puv - p)) < 1e-8:
                return u0, v0
        us = np.linspace(self.surface.u_range[0], self.surface.u_range[1], 64)
        vs = np.linspace(self.surface.v_range[0], self.surface.v_range[1], 64)
        U, V = np.meshgrid(us, vs)
        pts = self.surface.evaluate_local(U, V).reshape(-1, 3)
        dist = np.linalg.norm(pts - p, axis=1)
        valid = self._grid_mask(U, V, pts) & np.isfinite(dist)
        if not valid.any():
            return None
        idx = int(np.argmin(np.where(valid, dist, np.inf)))
        u = float(U.ravel()[idx])
        v = float(V.ravel()[idx])
        return self._solve_parameters_point(p, u, v)

    def parameters_at_xy(self, x, y):
        x = float(x)
        y = float(y)

        us = np.linspace(self.surface.u_range[0], self.surface.u_range[1], 64)
        vs = np.linspace(self.surface.v_range[0], self.surface.v_range[1], 64)

        U, V = np.meshgrid(us, vs)
        pts = self.surface.evaluate_local(U, V)

        finite = np.all(np.isfinite(pts), axis=-1)

        if not finite.any():
            return None

        distance2 = np.where(finite, ((pts[..., 0] - x)**2 + (pts[..., 1] - y)**2), np.inf)
        order = np.argsort(distance2.ravel())

        for idx in order[:12]:
            i, j = np.unravel_index(idx, U.shape)

            result = self._solve_parameters_xy(x, y, float(U[i, j]), float(V[i, j]))

            if result is not None:
                return result

        return None

    def intersect(self, ray):
        o = np.asarray(ray.origin, dtype=float)
        d = np.asarray(ray.direction, dtype=float)
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
            return None
        d_norm = float(np.linalg.norm(d))
        if not np.isfinite(d_norm) or d_norm <= 0.0:
            return None
        ray = Ray(o, d / d_norm)
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
                if self.surface.contains_point(u, v):
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
                    and self.surface.contains_point(u, v)
                ):
                    return float(u), float(v)
                return None
        puv = self.surface.evaluate_local(u, v)
        F = puv - p
        if (
            np.all(np.isfinite(F))
            and float(np.linalg.norm(F)) < 1e-8
            and self.surface.contains_point(u, v)
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
                if self.surface.contains_point(u, v):
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
                    and self.surface.contains_point(u, v)
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
            and self.surface.contains_point(u, v)
        ):
            return float(u), float(v)

        return None

    def _grid_mask(self, U, V, pts):
        finite = np.all(np.isfinite(pts), axis=1)
        domain = np.array([
            bool(self.surface.contains_point(u, v))
            for u, v in zip(U.ravel(), V.ravel())
        ])
        return finite & domain

    def _intersection_seeds(self, ray, grid_n=64, seed_tol=0.75, max_seeds=12):
        us = np.linspace(self.surface.u_range[0], self.surface.u_range[1], grid_n)
        vs = np.linspace(self.surface.v_range[0], self.surface.v_range[1], grid_n)
        U, V = np.meshgrid(us, vs)
        pts = self.surface.evaluate_local(U, V).reshape(-1, 3)
        valid_flat = self._grid_mask(U, V, pts)
        if not valid_flat.any():
            return []
        Uf = U.ravel()[valid_flat]
        Vf = V.ravel()[valid_flat]
        P = self.surface.transform.points_to_world(pts[valid_flat])
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
            and self.surface.contains_point(u, v)
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
