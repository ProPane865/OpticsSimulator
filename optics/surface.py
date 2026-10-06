import numpy as np
import sympy as sp

from .aperture import Aperture
from .solver import SurfaceSolver
from .transform import Transform

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
        if transform is None:
            self.transform = Transform(np.eye(3, 3), np.zeros(3))
        elif isinstance(transform, Transform):
            self.transform = transform
        else:
            self.transform = Transform(np.asarray(transform, dtype=float), np.zeros(3))
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
        self.solver = SurfaceSolver(self)

    @property
    def parameter_range(self):
        return (self.u_range, self.v_range)

    def point(self, u, v):
        return self.transform.point_to_world(self.evaluate_local(u, v))

    def tangents(self, u, v):
        ru, rv = self.derivatives_local(u, v)
        return self.transform.vector_to_world(ru), self.transform.vector_to_world(rv)

    def _inside_aperture(self, u, v):
        if self.aperture is None:
            return True

        return self.aperture.contains(self, u, v)

    def parameters_at(self, pos):
        return self.solver.parameters_at(pos)

    def parameters_at_xy(self, x, y):
        return self.solver.parameters_at_xy(x, y)

    def normal_at(self, pos):
        uv = self.parameters_at(pos)
        if uv is None:
            return None
        return self.normal(uv[0], uv[1])

    def aperture_mask(self, U, V, X, Y, Z):
        finite = (
            np.isfinite(X)
            & np.isfinite(Y)
            & np.isfinite(Z)
        )

        if self.aperture is None:
            return finite

        return self.aperture.mask(self, U, V, X, Y, Z)

    def intersect(self, ray):
        return self.solver.intersect(ray)

    def evaluate(self, u, v):
        """World-space point."""
        return self.transform.point_to_world(self.evaluate_local(u, v))

    def evaluate_local(self, U, V):
        """Local-space point (can be array)."""
        U, V = np.broadcast_arrays(U, V)
        shape = U.shape
        with np.errstate(invalid="ignore", divide="ignore"):
            X = np.asarray(self._x(U, V), dtype=float)
            Y = np.asarray(self._y(U, V), dtype=float)
            Z = np.asarray(self._z_eval(U, V), dtype=float)
        
        X = np.broadcast_to(X, shape)
        Y = np.broadcast_to(Y, shape)
        Z = np.broadcast_to(Z, shape)

        return np.stack((X, Y, Z), axis=-1)

    def derivatives_local(self, u, v):
        """Return du and dv tangent vectors."""
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

    def normal(self, u, v):
        """World-space unit normal."""
        if not self.accepts_parameters(u, v):
            return None

        ru, rv = self.derivatives_local(u, v)
        n = -np.cross(ru, rv)
        norm = np.linalg.norm(n)
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        return self.transform.vector_to_world(n / norm)

    def contains_parameters(self, u, v):
        """Whether (u,v) is in the surface domain."""
        tu = 1e-9 * max(1.0, abs(self.u_range[1] - self.u_range[0]))
        tv = 1e-9 * max(1.0, abs(self.v_range[1] - self.v_range[0]))

        return (
            self.u_range[0] - tu <= u <= self.u_range[1] + tu
            and self.v_range[0] - tv <= v <= self.v_range[1] + tv
        )

    def accepts_parameters(self, u, v):
        """Includes aperture restrictions."""
        if not self.contains_parameters(u, v):
            return False

        return self._inside_aperture(u, v)