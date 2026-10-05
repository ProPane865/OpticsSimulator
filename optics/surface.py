import numpy as np
import sympy as sp

from .aperture import Aperture, CircularAperture
from .intersection import SurfaceIntersect

def _clamp_sqrt_arguments(expr):
    replacements = {}
    for node in expr.atoms(sp.Pow):
        if node.exp == sp.Rational(1, 2):
            replacements[node] = sp.sqrt(sp.Max(node.base, 0))
    return expr.xreplace(replacements)

class LensSidewall:
    def __init__(self, front, rear):
        self.front = np.array(front, dtype=float)
        self.rear = np.array(rear, dtype=float)

        if self.front.shape != self.rear.shape:
            raise ValueError("Front and rear rims must correspond")

        if self.front.ndim != 2 or self.front.shape[1] != 3:
            raise ValueError("Rims must have shape (N, 3)")

        if len(self.front) < 3:
            raise ValueError("Rims must have at least 3 points")

        if not (
            np.all(np.isfinite(self.front))
            and np.all(np.isfinite(self.rear))
        ):
            raise ValueError("Rim points must be finite")

        n = len(self.front)
        self._verts = np.vstack([self.front, self.rear])

        i = np.arange(n)
        j = (i + 1) % n
        f0, f1 = i, j
        r0, r1 = n + i, n + j

        self._faces = np.empty((2 * n, 3), dtype=np.int64)
        self._faces[0::2] = np.column_stack([f0, r0, r1])
        self._faces[1::2] = np.column_stack([f0, r1, f1])

    def mesh(self):
        return self._verts, self._faces

    def intersect(self, ray):
        o = np.asarray(ray.origin, dtype=float)
        d = np.asarray(ray.direction, dtype=float)

        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
            return None

        d_norm = float(np.linalg.norm(d))

        if not np.isfinite(d_norm) or d_norm <= 0.0:
            return None

        d = d / d_norm

        tri = self._verts[self._faces]
        a = tri[:, 0]
        e1 = tri[:, 1] - a
        e2 = tri[:, 2] - a

        h = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, h)

        candidate = np.abs(det) > 1e-14

        inv_det = np.zeros_like(det)
        inv_det[candidate] = 1.0 / det[candidate]

        s = o - a
        u = np.einsum("ij,ij->i", s, h) * inv_det
        q = np.cross(s, e1)
        v = (q @ d) * inv_det
        t = np.einsum("ij,ij->i", e2, q) * inv_det

        tol = 1e-12

        hit = (
            candidate
            & (t > 1e-9)
            & (u >= -tol)
            & (v >= -tol)
            & (u + v <= 1.0 + tol)
        )

        if not hit.any():
            return None

        k = int(np.argmin(np.where(hit, t, np.inf)))

        u_k = min(max(float(u[k]), 0.0), 1.0)
        v_k = min(max(float(v[k]), 0.0), 1.0 - u_k)

        p = o + float(t[k]) * d

        return p, u_k, v_k

class Surface:
    def __init__(
        self,
        x_expr,
        y_expr,
        z_expr,
        transform=None,
        u_range=(-2.0, 2.0),
        v_range=(-2.0, 2.0),
        aperture=None
    ):
        self.u = sp.symbols("u")
        self.v = sp.symbols("v")
        self.x_expr = sp.parse_expr(str(x_expr))
        self.y_expr = sp.parse_expr(str(y_expr))
        self.z_expr = sp.parse_expr(str(z_expr))
        self.transform = np.eye(3, 3) if transform is None else np.asarray(transform, dtype=float)
        self.u_range = (float(u_range[0]), float(u_range[1]))
        self.v_range = (float(v_range[0]), float(v_range[1]))
        self.aperture = Aperture.from_spec(aperture)
        self._x = sp.lambdify((self.u, self.v), self.x_expr, modules=["numpy"])
        self._y = sp.lambdify((self.u, self.v), self.y_expr, modules=["numpy"])
        self._z = sp.lambdify((self.u, self.v), self.z_expr, modules=["numpy"])
        if self.aperture is not None:
            self._z_eval = sp.lambdify(
                (self.u, self.v),
                _clamp_sqrt_arguments(self.z_expr),
                modules=["numpy"],
            )
        else:
            self._z_eval = self._z
        self._rx = sp.lambdify((self.u, self.v), sp.diff(self.x_expr, self.u), modules=["numpy"])
        self._ry = sp.lambdify((self.u, self.v), sp.diff(self.y_expr, self.u), modules=["numpy"])
        self._rz = sp.lambdify((self.u, self.v), sp.diff(self.z_expr, self.u), modules=["numpy"])
        self._sx = sp.lambdify((self.u, self.v), sp.diff(self.x_expr, self.v), modules=["numpy"])
        self._sy = sp.lambdify((self.u, self.v), sp.diff(self.y_expr, self.v), modules=["numpy"])
        self._sz = sp.lambdify((self.u, self.v), sp.diff(self.z_expr, self.v), modules=["numpy"])
        self.intersector = SurfaceIntersect(self)

    @property
    def parameter_range(self):
        return (self.u_range, self.v_range)

    def _point_local(self, u, v):
        u = float(u)
        v = float(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            x = float(self._x(u, v))
            y = float(self._y(u, v))
            z = float(self._z_eval(u, v))
        return np.array([x, y, z], dtype=float)

    def _point_local_array(self, U, V):
        shape = np.shape(U)
        with np.errstate(invalid="ignore", divide="ignore"):
            X = np.asarray(self._x(U, V), dtype=float)
            Y = np.asarray(self._y(U, V), dtype=float)
            Z = np.asarray(self._z_eval(U, V), dtype=float)
        return (
            np.broadcast_to(X, shape),
            np.broadcast_to(Y, shape),
            np.broadcast_to(Z, shape),
        )

    def _tangent_local(self, u, v):
        u = float(u)
        v = float(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            ru = np.array(
                [
                    float(self._rx(u, v)),
                    float(self._ry(u, v)),
                    float(self._rz(u, v)),
                ],
                dtype=float,
            )
            rv = np.array(
                [
                    float(self._sx(u, v)),
                    float(self._sy(u, v)),
                    float(self._sz(u, v)),
                ],
                dtype=float,
            )
        return ru, rv

    def point(self, u, v):
        return self.transform @ self._point_local(u, v)

    def tangents(self, u, v):
        ru, rv = self._tangent_local(u, v)
        return self.transform @ ru, self.transform @ rv

    def normal(self, u, v):
        if not self._parameters_in_domain(u, v):
            return None

        ru, rv = self._tangent_local(u, v)
        n = -np.cross(ru, rv)
        norm = np.linalg.norm(n)
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        return self.transform @ (n / norm)

    def _parameters_in_domain(self, u, v):
        tu = 1e-9 * max(1.0, abs(self.u_range[1] - self.u_range[0]))
        tv = 1e-9 * max(1.0, abs(self.v_range[1] - self.v_range[0]))

        in_parameter_range = (
            self.u_range[0] - tu <= u <= self.u_range[1] + tu
            and self.v_range[0] - tv <= v <= self.v_range[1] + tv
        )

        if not in_parameter_range:
            return False

        return self._inside_aperture(u, v)

    def _inside_aperture(self, u, v):
        if self.aperture is None:
            return True

        return self.aperture.contains(self, u, v)

    def parameters_at(self, pos):
        p = self.transform.T @ np.asarray(pos, dtype=float)
        if not np.all(np.isfinite(p)):
            return None
        u0 = float(p[0])
        v0 = float(p[1])
        if self._parameters_in_domain(u0, v0):
            puv = self._point_local(u0, v0)
            if np.all(np.isfinite(puv)) and float(np.linalg.norm(puv - p)) < 1e-8:
                return u0, v0
        us = np.linspace(self.u_range[0], self.u_range[1], 64)
        vs = np.linspace(self.v_range[0], self.v_range[1], 64)
        U, V = np.meshgrid(us, vs)
        X, Y, Z = self._point_local_array(U, V)
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        dist = np.linalg.norm(pts - p, axis=1)
        valid = (
            self._aperture_mask(U, V, X, Y, Z).ravel()
            & np.isfinite(dist)
        )
        if not valid.any():
            return None
        idx = int(np.argmin(np.where(valid, dist, np.inf)))
        u = float(U.ravel()[idx])
        v = float(V.ravel()[idx])
        return self.intersector._solve_parameters_point(p, u, v)

    def parameters_at_xy(self, x, y):
        x = float(x)
        y = float(y)

        us = np.linspace(self.u_range[0], self.u_range[1], 64)
        vs = np.linspace(self.v_range[0], self.v_range[1], 64)

        U, V = np.meshgrid(us, vs)
        X, Y, Z = self._point_local_array(U, V)

        finite = (
            np.isfinite(X)
            & np.isfinite(Y)
            & np.isfinite(Z)
        )

        if not finite.any():
            return None

        distance2 = np.where(finite, ((X - x)**2 + (Y - y)**2), np.inf)
        order = np.argsort(distance2.ravel())

        for idx in order[:12]:
            i, j = np.unravel_index(idx, U.shape)

            result = self.intersector._solve_parameters_xy(x, y, float(U[i, j]), float(V[i, j]))

            if result is not None:
                return result

        return None

    def normal_at(self, pos):
        uv = self.parameters_at(pos)
        if uv is None:
            return None
        return self.normal(uv[0], uv[1])

    def _aperture_mask(self, U, V, X, Y, Z):
        finite = (
            np.isfinite(X)
            & np.isfinite(Y)
            & np.isfinite(Z)
        )

        if self.aperture is None:
            return finite

        return self.aperture.mask(self, U, V, X, Y, Z)


    def intersect(self, ray):
        return self.intersector.intersect(ray)