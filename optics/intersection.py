import numpy as np

class SurfaceIntersect:
    def __init__(self, surface):
        self.surface = surface

    def _solve_parameters_point(self, p, u, v):
        for _ in range(50):
            puv = self.surface._point_local(u, v)
            F = puv - p
            if not np.all(np.isfinite(F)):
                return None
            f_norm = float(np.linalg.norm(F))
            if f_norm < 1e-11:
                if self.surface._parameters_in_domain(u, v):
                    return float(u), float(v)
                return None
            ru, rv = self.surface._tangent_local(u, v)
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
                puv = self.surface._point_local(u, v)
                F = puv - p
                if (
                    np.all(np.isfinite(F))
                    and float(np.linalg.norm(F)) < 1e-9
                    and self.surface._parameters_in_domain(u, v)
                ):
                    return float(u), float(v)
                return None
        puv = self.surface._point_local(u, v)
        F = puv - p
        if (
            np.all(np.isfinite(F))
            and float(np.linalg.norm(F)) < 1e-8
            and self.surface._parameters_in_domain(u, v)
        ):
            return float(u), float(v)
        return None

    def _solve_parameters_xy(self, x_target, y_target, u, v):
        for _ in range(50):
            p = self.surface._point_local(u, v)

            if not np.all(np.isfinite(p)):
                return None

            F = np.array([
                p[0] - x_target,
                p[1] - y_target
            ])

            f_norm = float(np.linalg.norm(F))

            if f_norm < 1e-11:
                if self.surface._parameters_in_domain(u, v):
                    return float(u), float(v)
                return None

            ru, rv = self.surface._tangent_local(u, v)

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
                p = self.surface._point_local(u, v)
                F = np.array([
                    p[0] - x_target,
                    p[1] - y_target
                ])
                if (
                    np.all(np.isfinite(F))
                    and float(np.linalg.norm(F)) < 1e-9
                    and self.surface._parameters_in_domain(u, v)
                ):
                    return float(u), float(v)
                return None

        p = self.surface._point_local(u, v)

        if not np.all(np.isfinite(p)):
            return None

        F = np.array([
            p[0] - x_target,
            p[1] - y_target
        ])

        if (
            np.all(np.isfinite(F))
            and float(np.linalg.norm(F)) < 1e-8
            and self.surface._parameters_in_domain(u, v)
        ):
            return float(u), float(v)

        return None

    def _intersection_seeds(self, o, d, grid_n=64, seed_tol=0.75, max_seeds=12):
        us = np.linspace(self.surface.u_range[0], self.surface.u_range[1], grid_n)
        vs = np.linspace(self.surface.v_range[0], self.surface.v_range[1], grid_n)
        U, V = np.meshgrid(us, vs)
        X, Y, Z = self.surface._point_local_array(U, V)
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        valid = self.surface._aperture_mask(U, V, X, Y, Z)
        valid_flat = valid.ravel()
        if not valid_flat.any():
            return []
        Uf = U.ravel()[valid_flat]
        Vf = V.ravel()[valid_flat]
        P = pts[valid_flat]
        rel = P - o
        t = rel @ d
        forward = t > 0.0
        if not forward.any():
            return []
        proj = o + t[:, None] * d
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

    def _accept_intersection(self, o, d, t, u, v, tol):
        p = self.surface._point_local(u, v)
        F = p - (o + t * d)
        if (
            np.all(np.isfinite(F))
            and float(np.linalg.norm(F)) < tol
            and t > 0.0
            and self.surface._parameters_in_domain(u, v)
        ):
            return self.surface.transform @ (o + t * d), float(u), float(v)
        return None

    def _newton_intersection(self, o, d, u, v, t):
        for _ in range(50):
            p = self.surface._point_local(u, v)
            F = p - (o + t * d)
            if not np.all(np.isfinite(F)):
                return None
            if float(np.linalg.norm(F)) < 1e-12:
                return self._accept_intersection(o, d, t, u, v, 1e-11)
            ru, rv = self.surface._tangent_local(u, v)
            if not (np.all(np.isfinite(ru)) and np.all(np.isfinite(rv))):
                return None
            A = np.column_stack((-d, ru, rv))
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
                return self._accept_intersection(o, d, t, u, v, 1e-10)
        return self._accept_intersection(o, d, t, u, v, 1e-10)

    def intersect(self, ray):
        o = self.surface.transform.T @ np.asarray(ray.origin, dtype=float)
        d = self.surface.transform.T @ np.asarray(ray.direction, dtype=float)
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
            return None
        d_norm = float(np.linalg.norm(d))
        if not np.isfinite(d_norm) or d_norm <= 0.0:
            return None
        d = d / d_norm
        for u0, v0, t0 in self._intersection_seeds(o, d):
            hit = self._newton_intersection(o, d, u0, v0, t0)
            if hit is not None:
                return hit
        return None
