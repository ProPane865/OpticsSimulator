import functools
from collections.abc import Sequence
from typing import Any

import numpy as np
import sympy as sp

from . import constants as _c
from .aperture import Aperture
from .hit import Hit
from .ray import Ray
from .solver import SurfaceSolver
from .transform import Transform


@functools.lru_cache(maxsize=64)
def _compile_parametric_surface(x_src: str, y_src: str, z_src: str) -> tuple[Any, ...]:
    u = sp.symbols("u")
    v = sp.symbols("v")
    x_expr = sp.parse_expr(x_src)
    y_expr = sp.parse_expr(y_src)
    z_expr = sp.parse_expr(z_src)

    def lam(expr: sp.Expr) -> Any:
        return sp.lambdify((u, v), expr, modules=["numpy"])

    return (
        x_expr,
        y_expr,
        z_expr,
        lam(x_expr),
        lam(y_expr),
        lam(z_expr),
        lam(sp.diff(x_expr, u)),
        lam(sp.diff(y_expr, u)),
        lam(sp.diff(z_expr, u)),
        lam(sp.diff(x_expr, v)),
        lam(sp.diff(y_expr, v)),
        lam(sp.diff(z_expr, v)),
    )

class Surface:
    def __init__(
        self,
        x_expr: Any,
        y_expr: Any,
        z_expr: Any,
        transform: "Transform | np.ndarray | None" = None,
        u_range: "Sequence[float]" = (-2.0, 2.0),
        v_range: "Sequence[float]" = (-2.0, 2.0),
        aperture: "Aperture | dict[str, Any] | None" = None,
    ) -> None:
        (
            self.x_expr,
            self.y_expr,
            self.z_expr,
            self._x,
            self._y,
            self._z,
            self._rx,
            self._ry,
            self._rz,
            self._sx,
            self._sy,
            self._sz,
        ) = _compile_parametric_surface(str(x_expr), str(y_expr), str(z_expr))
        if transform is None:
            self.transform = Transform(np.eye(3, 3), np.zeros(3))
        elif isinstance(transform, Transform):
            self.transform = transform
        else:
            self.transform = Transform(np.asarray(transform, dtype=float), np.zeros(3))
        self.u_range = (float(u_range[0]), float(u_range[1]))
        self.v_range = (float(v_range[0]), float(v_range[1]))
        self.aperture = Aperture.from_spec(aperture)
        self.solver = SurfaceSolver(self)

    @property
    def parameter_range(self) -> "tuple[tuple[float, float], tuple[float, float]]":
        return (self.u_range, self.v_range)

    def point(self, u: float, v: float) -> np.ndarray:
        return self.transform.point_to_world(self.evaluate_local(u, v))

    def tangents(self, u: float, v: float) -> "tuple[np.ndarray, np.ndarray]":
        ru, rv = self.derivatives_local(u, v)
        return self.transform.vector_to_world(ru), self.transform.vector_to_world(rv)

    def _inside_aperture(self, u: float, v: float) -> bool:
        if self.aperture is None:
            return True

        return self.aperture.contains(self, u, v)

    def parameters_at(self, pos: np.ndarray | Sequence[float]) -> "tuple[float, float] | None":
        return self.solver.parameters_at(pos)

    def parameters_at_xy(
        self,
        x: float,
        y: float,
        seed: "tuple[float, float] | None" = None,
    ) -> "tuple[float, float] | None":
        return self.solver.parameters_at_xy(x, y, seed=seed)

    def normal_at(self, pos: np.ndarray | Sequence[float]) -> np.ndarray | None:
        uv = self.parameters_at(pos)
        if uv is None:
            return None
        return self.normal(uv[0], uv[1])

    def aperture_mask(
        self,
        U: np.ndarray,
        V: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        finite = (
            np.isfinite(X)
            & np.isfinite(Y)
            & np.isfinite(Z)
        )

        if self.aperture is None:
            return finite

        return self.aperture.mask(self, U, V, X, Y, Z)

    def intersect(self, ray: Ray) -> Hit | None:
        return self.solver.intersect(ray)

    def evaluate(self, u: float, v: float) -> np.ndarray:
        """World-space point."""
        return self.transform.point_to_world(self.evaluate_local(u, v))

    def evaluate_local(self, U: Any, V: Any) -> np.ndarray:
        """Local-space point (can be array)."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            if U.ndim == 0 and V.ndim == 0:
                u = float(U)
                v = float(V)
                return np.array(
                    [self._x(u, v), self._y(u, v), self._z(u, v)],
                    dtype=float,
                )
            U, V = np.broadcast_arrays(U, V)
            shape = U.shape
            X = np.asarray(self._x(U, V), dtype=float)
            Y = np.asarray(self._y(U, V), dtype=float)
            Z = np.asarray(self._z(U, V), dtype=float)

        X = np.broadcast_to(X, shape)
        Y = np.broadcast_to(Y, shape)
        Z = np.broadcast_to(Z, shape)

        return np.stack((X, Y, Z), axis=-1)

    def derivatives_local(self, u: float, v: float) -> "tuple[np.ndarray, np.ndarray]":
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

    def normal(self, u: float, v: float) -> np.ndarray | None:
        """World-space unit normal."""
        if not self.accepts_parameters(u, v):
            return None

        ru, rv = self.derivatives_local(u, v)
        n = -np.cross(ru, rv)
        norm = np.linalg.norm(n)
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        return self.transform.vector_to_world(n / norm)

    def in_domain(self, u: Any, v: Any) -> Any:
        """Vectorized check whether parameters (u, v) lie in the domain."""
        tu = _c.DOMAIN_TOL * max(1.0, abs(self.u_range[1] - self.u_range[0]))
        tv = _c.DOMAIN_TOL * max(1.0, abs(self.v_range[1] - self.v_range[0]))

        return (
            (u >= self.u_range[0] - tu)
            & (u <= self.u_range[1] + tu)
            & (v >= self.v_range[0] - tv)
            & (v <= self.v_range[1] + tv)
        )

    def contains_parameters(self, u: float, v: float) -> bool:
        """Whether (u,v) is in the surface domain."""
        return bool(self.in_domain(float(u), float(v)))

    def accepts_parameters(self, u: float, v: float) -> bool:
        """Includes aperture restrictions."""
        if not self.contains_parameters(u, v):
            return False

        return self._inside_aperture(u, v)