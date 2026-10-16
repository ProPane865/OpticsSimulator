"""Test-only, pure-Python reference surface and tracing oracle.

The production package runs everything through Numba kernels. This module keeps
an *independent* implementation of the public :class:`optics.primitive.Surface`
API -- evaluation, finite-difference tangents, and the damped-Newton
intersection / parameter-inversion solvers -- so the differential parity tests
can exercise the nopython kernels against a reference that shares none of their
flat-tuple/index code. It is imported only by tests.

Evaluation reuses the validated AST lowering from
:mod:`optics.numerical.expression` and ``math``; derivatives reuse
:mod:`optics.numerical.differentiation`; Newton reuses the pure-Python
:func:`optics.numerical.newton._newton`. The solver mirrors the kernels'
seed-and-refine structure (grid seeds, forward hit, dedup) in plain Python.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np

from optics import constants as _c
from optics.numerical import expression
from optics.numerical.differentiation import numerical_derivatives
from optics.numerical.newton import _newton
from optics.primitive import Hit, Transform
from optics.tracing.element import _SIDEWALL_UNSET


def _sdiv(a: float, b: float) -> float:
    if b == 0.0:
        if a == 0.0:
            return math.nan
        return math.copysign(math.inf, a)
    return a / b


def _smod(a: float, b: float) -> float:
    if b == 0.0:
        return math.nan
    return a % b


def _sqrt(x: float) -> float:
    # Match Numba's ``math.sqrt`` (nan outside the domain), not Python's raise.
    return math.sqrt(x) if x >= 0.0 else math.nan


def _log(x: float) -> float:
    if x > 0.0:
        return math.log(x)
    if x == 0.0:
        return -math.inf
    return math.nan


def _exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


# ``math`` namespace mirroring the generated Numba module's scalar semantics.
_MATH = SimpleNamespace(sqrt=_sqrt, log=_log, exp=_exp, sin=math.sin, cos=math.cos, tan=math.tan)
_NS = {"math": _MATH, "_sdiv": _sdiv, "_smod": _smod}


def _compile(source: str) -> Any:
    code = compile(expression.scalar_source(source), "<expr>", "eval")
    return lambda u, v: eval(code, _NS, {"u": u, "v": v})


class ReferenceSurface:
    """Pure-Python mirror of :class:`~optics.primitive.Surface` (no Numba)."""

    def __init__(
        self,
        x_expr: Any,
        y_expr: Any,
        z_expr: Any,
        transform: Transform | np.ndarray | None = None,
        u_range: Sequence[float] = (-2.0, 2.0),
        v_range: Sequence[float] = (-2.0, 2.0),
        aperture: Any = None,
    ) -> None:
        self.x_expr = str(x_expr)
        self.y_expr = str(y_expr)
        self.z_expr = str(z_expr)
        self._fx = _compile(self.x_expr)
        self._fy = _compile(self.y_expr)
        self._fz = _compile(self.z_expr)
        if transform is None:
            self.transform = Transform(np.eye(3, 3), np.zeros(3))
        elif isinstance(transform, Transform):
            self.transform = transform
        else:
            self.transform = Transform(np.asarray(transform, dtype=float), np.zeros(3))
        self.u_range = (float(u_range[0]), float(u_range[1]))
        self.v_range = (float(v_range[0]), float(v_range[1]))
        self.aperture = aperture
        self._grid_cache: dict[int, dict[str, np.ndarray]] | None = None
        self._graph_xy: bool | None = None
        self._njit_desc_built = False

    # -- evaluation ----------------------------------------------------------
    def _point(self, u: float, v: float) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.array([self._fx(u, v), self._fy(u, v), self._fz(u, v)], dtype=float)

    def evaluate_local(self, U: Any, V: Any) -> np.ndarray:
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        if U.ndim == 0 and V.ndim == 0:
            return self._point(float(U), float(V))
        U, V = np.broadcast_arrays(U, V)
        shape = U.shape
        flat = np.array([self._point(float(u), float(v)) for u, v in zip(U.ravel(), V.ravel())])
        return flat.reshape(shape + (3,))

    # -- tangents ------------------------------------------------------------
    def _local_tangents(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        return numerical_derivatives(self._point, float(u), float(v), self.u_range, self.v_range)

    def _partial_local(self, u: float, v: float, du: float, dv: float) -> np.ndarray:
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

    def _interior_derivatives(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
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

    def derivatives_local(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        u = float(u)
        v = float(v)
        ru, rv = self._local_tangents(u, v)
        if not (np.all(np.isfinite(ru)) and np.all(np.isfinite(rv))):
            ru, rv = self._interior_derivatives(u, v)
        return ru, rv

    # -- domain / membership -------------------------------------------------
    @property
    def parameter_range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.u_range, self.v_range)

    @property
    def njit_descriptor(self) -> tuple[Any, ...] | None:
        # Reference surfaces always run on the pure-Python path.
        return None

    def in_domain(self, u: Any, v: Any) -> Any:
        tu = _c.DOMAIN_TOL * max(1.0, abs(self.u_range[1] - self.u_range[0]))
        tv = _c.DOMAIN_TOL * max(1.0, abs(self.v_range[1] - self.v_range[0]))
        return (
            (u >= self.u_range[0] - tu)
            & (u <= self.u_range[1] + tu)
            & (v >= self.v_range[0] - tv)
            & (v <= self.v_range[1] + tv)
        )

    def contains_parameters(self, u: float, v: float) -> bool:
        return bool(self.in_domain(float(u), float(v)))

    def _inside_aperture(self, u: float, v: float) -> bool:
        return True if self.aperture is None else self.aperture.contains(self, u, v)

    def accepts_parameters(self, u: float, v: float) -> bool:
        if not self.contains_parameters(u, v):
            return False
        return self._inside_aperture(u, v)

    @property
    def is_graph_over_xy(self) -> bool:
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

    # -- world-space helpers -------------------------------------------------
    def point(self, u: float, v: float) -> np.ndarray:
        return self.transform.point_to_world(self.evaluate_local(u, v))

    def tangents(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        ru, rv = self.derivatives_local(u, v)
        return self.transform.vector_to_world(ru), self.transform.vector_to_world(rv)

    def normal(self, u: float, v: float) -> np.ndarray | None:
        if not self.accepts_parameters(u, v):
            return None
        ru, rv = self.derivatives_local(u, v)
        n = -np.cross(ru, rv)
        norm = np.linalg.norm(n)
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        return self.transform.vector_to_world(n / norm)

    # -- seed grid -----------------------------------------------------------
    def _grid_mask(self, U: np.ndarray, V: np.ndarray, pts: np.ndarray) -> np.ndarray:
        finite = np.all(np.isfinite(pts), axis=1)
        uf = U.ravel()
        vf = V.ravel()
        mask = finite & self.in_domain(uf, vf)
        if self.aperture is not None:
            mask = mask & np.asarray(
                self.aperture.mask(self, uf, vf, pts[:, 0], pts[:, 1], pts[:, 2]),
                dtype=bool,
            )
        return mask

    def _seed_grid(self, grid_n: int = _c.SEED_GRID_N) -> dict[str, np.ndarray]:
        if self._grid_cache is not None and grid_n in self._grid_cache:
            return self._grid_cache[grid_n]
        (u_min, u_max), (v_min, v_max) = self.parameter_range
        us = np.linspace(u_min, u_max, grid_n)
        vs = np.linspace(v_min, v_max, grid_n)
        U, V = np.meshgrid(us, vs)
        Uf = U.ravel()
        Vf = V.ravel()
        pts = self.evaluate_local(U, V).reshape(-1, 3)
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
                self.transform.points_to_world(pts[valid])
                if valid.any()
                else np.empty((0, 3), dtype=float)
            ),
        }
        if self._grid_cache is None:
            self._grid_cache = {}
        self._grid_cache[grid_n] = entry
        return entry

    # -- parameter inversion (Newton) ---------------------------------------
    def _solve_parameters_point(
        self, p: np.ndarray, u: float, v: float
    ) -> tuple[float, float] | None:
        def residual(x: np.ndarray) -> np.ndarray:
            return self.evaluate_local(float(x[0]), float(x[1])) - p

        def jacobian(x: np.ndarray) -> np.ndarray:
            ru, rv = self.derivatives_local(float(x[0]), float(x[1]))
            return np.column_stack((ru, rv))

        result = _newton(residual, jacobian, (u, v))
        if result is None:
            return None
        x, f_norm, status = result
        if f_norm >= float(_c.PARAM_TOLS[status]):
            return None
        u = float(x[0])
        v = float(x[1])
        if not self.accepts_parameters(u, v):
            return None
        return u, v

    def _solve_parameters_xy(
        self, x_target: float, y_target: float, u: float, v: float
    ) -> tuple[float, float] | None:
        def residual(x: np.ndarray) -> np.ndarray:
            p = self.evaluate_local(float(x[0]), float(x[1]))
            return np.array([p[0] - x_target, p[1] - y_target])

        def jacobian(x: np.ndarray) -> np.ndarray:
            ru, rv = self.derivatives_local(float(x[0]), float(x[1]))
            return np.array([[ru[0], rv[0]], [ru[1], rv[1]]])

        result = _newton(residual, jacobian, (u, v))
        if result is None:
            return None
        x, f_norm, status = result
        if f_norm >= float(_c.PARAM_TOLS[status]):
            return None
        u = float(x[0])
        v = float(x[1])
        if not self.accepts_parameters(u, v):
            return None
        return u, v

    def parameters_at(self, pos: np.ndarray | Sequence[float]) -> tuple[float, float] | None:
        p = self.transform.point_to_local(np.asarray(pos, dtype=float))
        if not np.all(np.isfinite(p)):
            return None
        u0 = float(p[0])
        v0 = float(p[1])
        if self.accepts_parameters(u0, v0):
            puv = self.evaluate_local(u0, v0)
            if np.all(np.isfinite(puv)) and float(np.linalg.norm(puv - p)) < _c.POINT_MATCH_TOL:
                return u0, v0
        grid = self._seed_grid()
        dist = np.linalg.norm(grid["pts"] - p, axis=1)
        valid = grid["valid"] & np.isfinite(dist)
        if not valid.any():
            return None
        idx = int(np.argmin(np.where(valid, dist, np.inf)))
        return self._solve_parameters_point(p, float(grid["Uf"][idx]), float(grid["Vf"][idx]))

    def parameters_at_xy(
        self, x: float, y: float, seed: tuple[float, float] | None = None
    ) -> tuple[float, float] | None:
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
        distance2 = np.where(finite, ((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2), np.inf)
        k = min(_c.XY_SEEDS, distance2.size)
        order = np.argpartition(distance2, k - 1)[:k]
        order = order[np.argsort(distance2[order])]
        for idx in order:
            result = self._solve_parameters_xy(x, y, float(grid["Uf"][idx]), float(grid["Vf"][idx]))
            if result is not None:
                return result
        return None

    # -- ray/surface intersection (Newton) ----------------------------------
    def _newton_intersection(
        self, o: np.ndarray, d: np.ndarray, u: float, v: float, t: float
    ) -> tuple[float, float, float] | None:
        def residual(x: np.ndarray) -> np.ndarray:
            return self.point(float(x[1]), float(x[2])) - (o + float(x[0]) * d)

        def jacobian(x: np.ndarray) -> np.ndarray:
            ru, rv = self.tangents(float(x[1]), float(x[2]))
            return np.column_stack((-d, ru, rv))

        result = _newton(residual, jacobian, (t, u, v), clamp=True)
        if result is None:
            return None
        x, f_norm, status = result
        if f_norm >= float(_c.HIT_TOLS[status]):
            return None
        t, u, v = float(x[0]), float(x[1]), float(x[2])
        if t > 0.0 and self.accepts_parameters(u, v):
            return t, u, v
        return None

    def _intersection_seeds(
        self, o: np.ndarray, d: np.ndarray, max_seeds: int = _c.MAX_SEEDS
    ) -> list[tuple[float, float, float]]:
        grid = self._seed_grid()
        if not grid["valid"].any():
            return []
        P = grid["P_world"]
        t = (P - o) @ d
        forward = t > 0.0
        if not forward.any():
            return []
        proj = o + t[:, None] * d
        residual = np.linalg.norm(P - proj, axis=1)
        mask = forward & np.isfinite(residual) & (residual <= _c.SEED_TOL)
        if not mask.any():
            mask = forward & np.isfinite(residual) & (residual <= _c.SEED_TOL_FALLBACK)
            if not mask.any():
                return []
        indices = np.where(mask)[0]
        order = indices[np.argsort(residual[indices])]
        Uf = grid["Uf_valid"]
        Vf = grid["Vf_valid"]
        return [
            (float(Uf[idx]), float(Vf[idx]), float(t[idx])) for idx in order[:max_seeds]
        ]

    def _intersect_uv(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> tuple[float, float, float] | None:
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
            return None
        d_norm = float(np.linalg.norm(d))
        if d_norm == 0.0:
            return None
        d = d / d_norm
        hits: list[tuple[float, float, float]] = []
        for u0, v0, t0 in self._intersection_seeds(o, d):
            hit = self._newton_intersection(o, d, u0, v0, t0)
            if hit is None:
                continue
            t, u, v = hit
            for i, other in enumerate(hits):
                if abs(u - other[1]) < _c.DEDUP_UV_TOL and abs(v - other[2]) < _c.DEDUP_UV_TOL:
                    if t < other[0]:
                        hits[i] = hit
                    break
            else:
                hits.append(hit)
        if not hits:
            return None
        return min(hits, key=lambda h: h[0])

    def intersect(self, ray: Any) -> Hit | None:
        hit = self._intersect_uv(ray.origin, ray.direction)
        if hit is None:
            return None
        t, u, v = hit
        return Hit(ray.at(t), t, u, v)


def reference_surface(surface: Any) -> ReferenceSurface:
    """Build a pure-Python clone of a compiled ``Surface`` (same source/ranges)."""
    return ReferenceSurface(
        surface.x_expr,
        surface.y_expr,
        surface.z_expr,
        transform=surface.transform,
        u_range=surface.u_range,
        v_range=surface.v_range,
        aperture=surface.aperture,
    )


def force_python_path(elements: Sequence[Any]) -> None:
    """Swap each element's surfaces for pure-Python reference clones in place.

    Downstream code (``RefractiveElement.trace``/``first_hit``/``contains_point``,
    ``traverse_stack``, ``build_trace_context`` returning ``None``) then falls to
    the reference implementation, independent of the Numba kernels.
    """
    for element in elements:
        element.surface1 = reference_surface(element.surface1)
        element.surface2 = reference_surface(element.surface2)
        element._sidewall = _SIDEWALL_UNSET
        element._local_aabb = None
