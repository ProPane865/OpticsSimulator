import numpy as np
import sympy as sp

from .aperture import Aperture
from .intersection import SurfaceIntersect

def _clamp_sqrt_arguments(expr):
    replacements = {}
    for node in expr.atoms(sp.Pow):
        if node.exp == sp.Rational(1, 2):
            replacements[node] = sp.sqrt(sp.Max(node.base, 0))
    return expr.xreplace(replacements)

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