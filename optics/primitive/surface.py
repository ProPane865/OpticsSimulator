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
        self._eval_local = fns.eval_local
        self._eval_grid = fns.eval_grid
        self._local_tangents = fns.derivatives_local
        self._eval_cfn = fns.eval_cfn
        self._deriv_cfn = fns.deriv_cfn
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
        self._njit_desc: tuple[Any, ...] | None = None
        self._njit_desc_built = False

    @property
    def njit_descriptor(self) -> "tuple[Any, ...] | None":
        """Flat tuple describing this surface for the nopython kernels.

        ``None`` when the surface cannot run in nopython mode (expression not
        compilable, or an aperture type the kernels do not understand), in
        which case callers fall back to the Python solver path.
        """
        if not self._njit_desc_built:
            self._njit_desc_built = True
            self._njit_desc = self._build_njit_descriptor()
        return self._njit_desc

    def _build_njit_descriptor(self) -> "tuple[Any, ...] | None":
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

    def _partial_local(self, u: float, v: float, du: float, dv: float) -> np.ndarray:
        """One-sided finite-difference partial in direction ``(du, dv)``.

        The step is halved until the evaluation stays finite, so it still works
        right at the edge of the aperture rim where tangents diverge.
        """
        base = self.evaluate_local(u, v)
        if not np.all(np.isfinite(base)):
            return np.full(3, np.nan)
        step = _c.FD_STEP_REL * max(1.0, abs(u), abs(v))
        for sign in (1.0, -1.0):
            h = step
            for _ in range(_c.FD_STEP_SHRINKS):
                p = self.evaluate_local(u + sign * du * h, v + sign * dv * h)
                if np.all(np.isfinite(p)):
                    return (p - base) / (sign * h)
                h *= 0.5
        return np.full(3, np.nan)

    def _interior_derivatives(self, u: float, v: float) -> "tuple[np.ndarray, np.ndarray]":
        """Tangents at the nearest interior point, used where they blow up.

        At an aperture rim a graph surface becomes vertical and a finite-difference
        sample crosses into an undefined region. Pulling back toward the interior
        of the parameter domain and evaluating there gives the limiting tangent
        directions, which is the geometrically correct rim frame.
        """
        (u_min, u_max), (v_min, v_max) = self.parameter_range
        du = 0.5 * (u_min + u_max) - u
        dv = 0.5 * (v_min + v_max) - v
        if du == 0.0 and dv == 0.0:
            return self._partial_local(u, v, 1.0, 0.0), self._partial_local(u, v, 0.0, 1.0)
        span = max(abs(u_max - u_min), abs(v_max - v_min), 1.0)
        length = float(np.hypot(du, dv))
        frac = _c.DERIV_INTERIOR_FRAC * span / length
        for _ in range(_c.DERIV_INTERIOR_TRIES):
            if frac >= 1.0:
                break
            uu = u + du * frac
            vv = v + dv * frac
            if self.accepts_parameters(uu, vv):
                ru, rv = self._local_tangents(uu, vv)
                if np.all(np.isfinite(ru)) and np.all(np.isfinite(rv)):
                    return ru, rv
            frac *= 2.0
        return self._partial_local(u, v, 1.0, 0.0), self._partial_local(u, v, 0.0, 1.0)

    def derivatives_local(self, u: float, v: float) -> "tuple[np.ndarray, np.ndarray]":
        """Return du and dv tangent vectors (finite differences)."""
        u = float(u)
        v = float(v)
        ru, rv = self._local_tangents(u, v)
        if not (np.all(np.isfinite(ru)) and np.all(np.isfinite(rv))):
            ru, rv = self._interior_derivatives(u, v)
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
