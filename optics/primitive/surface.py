from collections.abc import Sequence
from typing import Any

import numpy as np

from .. import constants as _c
from ..numerical.codegen import compile_parametric_surface
from ..numerical.desc import build_surface_descriptor
from ..numerical.solver import SurfaceSolver
from .aperture import Aperture
from .hit import Hit
from .ray import Ray
from .transform import Transform


class Surface:
    def __init__(
        self,
        x_expr: Any,
        y_expr: Any,
        z_expr: Any,
        transform: "Transform | np.ndarray | None" = None,
        u_range: "Sequence[float]" = (-2.0, 2.0),
        v_range: "Sequence[float]" = (-2.0, 2.0),
        aperture: "Aperture | None" = None,
    ) -> None:
        u_range = (float(u_range[0]), float(u_range[1]))
        v_range = (float(v_range[0]), float(v_range[1]))
        fns = compile_parametric_surface(
            str(x_expr), str(y_expr), str(z_expr), u_range, v_range
        )
        self.x_expr = fns.x_src
        self.y_expr = fns.y_src
        self.z_expr = fns.z_src
        self._model = fns.model
        self._eval_local = fns.eval_local
        self._eval_grid = fns.eval_grid
        if transform is None:
            self.transform = Transform(np.eye(3, 3), np.zeros(3))
        elif isinstance(transform, Transform):
            self.transform = transform
        else:
            self.transform = Transform(np.asarray(transform, dtype=float), np.zeros(3))
        self.u_range = u_range
        self.v_range = v_range
        self.aperture = aperture
        self.solver = SurfaceSolver(self)
        self._graph_xy: bool | None = None
        self._native_desc: tuple[Any, ...] | None = None
        self._native_desc_built = False

    @property
    def native_descriptor(self) -> Any:
        """Native descriptor for this surface, used by the kernel paths.

        Built lazily and cached; building raises :class:`ValueError` when the
        surface carries an aperture the kernels cannot encode, in which case
        the surface cannot take part in a native trace.
        """
        if not self._native_desc_built:
            self._native_desc_built = True
            self._native_desc = self._build_native_descriptor()
        return self._native_desc

    def _build_native_descriptor(self) -> Any:
        return build_surface_descriptor(self)

    @property
    def is_graph_over_xy(self) -> bool:
        """Whether the surface is exactly ``(u, v, z(u, v))`` in local space.

        Sampled once and cached; lets callers read the height above a local
        ``(x, y)`` without running the parameter-inversion solver.
        """
        if self._graph_xy is None:
            self._graph_xy = self._check_graph_over_xy()
        return self._graph_xy

    def _check_graph_over_xy(self) -> bool:
        (u_min, u_max), (v_min, v_max) = self.parameter_range
        for u in np.linspace(u_min, u_max, 5)[1:-1]:
            for v in np.linspace(v_min, v_max, 5)[1:-1]:
                u = float(u)
                v = float(v)
                p = self.evaluate_local(u, v)
                if not np.all(np.isfinite(p)):
                    return False
                if (
                    abs(float(p[0]) - u) > _c.GRAPH_XY_TOL * max(1.0, abs(u))
                    or abs(float(p[1]) - v) > _c.GRAPH_XY_TOL * max(1.0, abs(v))
                ):
                    return False
        return True

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
        hit = self.solver.intersect(ray.origin, ray.direction)
        if hit is None:
            return None
        t, u, v = hit
        return Hit(ray.at(t), t, u, v)

    def evaluate_local(self, U: Any, V: Any) -> np.ndarray:
        """Local-space point for scalar or broadcast array parameters."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        if U.ndim == 0 and V.ndim == 0:
            return self._eval_local(float(U), float(V))
        U, V = np.broadcast_arrays(U, V)
        shape = U.shape
        us = np.ascontiguousarray(U.reshape(-1), dtype=float)
        vs = np.ascontiguousarray(V.reshape(-1), dtype=float)
        return self._eval_grid(us, vs).reshape(shape + (3,))

    def derivatives_local(self, u: float, v: float) -> "tuple[np.ndarray, np.ndarray]":
        """Return the ``du``/``dv`` tangent vectors at ``(u, v)`` (local space).

        The finite-difference tangent, including the interior pull-back / rim
        fallback used where it blows up, runs natively on the surface descriptor.
        """
        ru, rv = self.native_descriptor.derivatives(float(u), float(v))
        return ru, rv

    def normal(self, u: float, v: float) -> np.ndarray | None:
        """World-space unit normal (``None`` outside the domain/aperture).

        Computed natively on the surface descriptor (the interior pull-back and
        the ``-cross(r_u, r_v)`` orientation with the world rotation).
        """
        n = self.native_descriptor.normal(float(u), float(v))
        return None if n is None else np.asarray(n, dtype=float)

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
