from collections.abc import Sequence
from typing import Any

import numpy as np

from .. import constants as _c
from ..numerical._kernels import make_surface
from ..numerical.codegen import compile_parametric_surface
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
        self._native: Any | None = None
        self._desc_handle: Any | None = None
        self._grid_handle: Any | None = None
        self._graph_xy: bool | None = None

    def _aperture_params(self) -> "tuple[float, float, float]":
        """Circular aperture slots ``(radius, cx, cy)`` for the native kernels.

        ``(-1, 0, 0)`` encodes "no aperture". Raises :class:`ValueError` when
        the surface carries an aperture the kernels cannot encode, so such a
        surface cannot take part in a native trace.
        """
        if self.aperture is None:
            return -1.0, 0.0, 0.0
        params = self.aperture.native_params()
        if params is None:
            raise ValueError(
                "Aperture type is not supported by the native surface kernels"
            )
        ap_r, ap_cx, ap_cy = params
        return float(ap_r), float(ap_cx), float(ap_cy)

    @property
    def native_surface(self) -> Any:
        """The persistent native surface (descriptor + seed grid + graph flag).

        Built lazily on first native use and cached for the surface's lifetime;
        the seed grid is sampled once, so no per-trace descriptor/grid work
        remains.
        """
        if self._native is None:
            ap_r, ap_cx, ap_cy = self._aperture_params()
            self._native = make_surface(
                self._model,
                self.x_expr,
                self.y_expr,
                self.z_expr,
                self.u_range[0],
                self.u_range[1],
                self.v_range[0],
                self.v_range[1],
                ap_r,
                ap_cx,
                ap_cy,
                self.transform.rotation,
                self.transform.translation,
                _c.SEED_GRID_N,
            )
        return self._native

    @property
    def native_descriptor(self) -> Any:
        """Native descriptor for this surface, used by the kernel paths.

        Owned by the persistent native surface; the Python handle is fetched
        once and cached so repeated kernel calls reuse one wrapper object.
        Building can raise :class:`ValueError` when the surface carries an
        aperture the kernels cannot encode.
        """
        if self._desc_handle is None:
            self._desc_handle = self.native_surface.desc()
        return self._desc_handle

    @property
    def native_grid(self) -> Any:
        """The persistent seed grid owned by the native surface."""
        if self._grid_handle is None:
            self._grid_handle = self.native_surface.grid()
        return self._grid_handle

    @property
    def is_graph_over_xy(self) -> bool:
        """Whether the surface is exactly ``(u, v, z(u, v))`` in local space.

        Sampled once by the native surface; lets callers read the height above
        a local ``(x, y)`` without running the parameter-inversion solver.
        """
        if self._graph_xy is None:
            self._graph_xy = bool(self.native_surface.is_graph_over_xy())
        return self._graph_xy

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
