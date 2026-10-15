"""Damped Newton iteration and its residual/Jacobian problems."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numba import njit

from .. import constants as _c
from ._kernels import D_ROT, D_TRANS
from .desc import derivatives_njit, eval_njit
from .linalg import _newton_delta, _newton_delta_njit, norm

_Residual = Callable[[np.ndarray], "np.ndarray | None"]
_Jacobian = Callable[[np.ndarray], "np.ndarray | None"]

# Solver outcome codes shared by the Python and nopython paths; they index the
# per-status tolerance tables in ``constants`` and are read inside the njit
# kernels as compile-time constants.
_CONVERGED = _c.SOLVER_CONV
_STALLED = _c.SOLVER_STALL
_EXHAUSTED = _c.SOLVER_EXH

_NEWTON_INTERSECT = 0
_NEWTON_XY = 1


def _newton(
    residual: _Residual,
    jacobian: _Jacobian,
    x0: Sequence[float],
    clamp: bool = False,
) -> "tuple[np.ndarray, float, int] | None":
    """Damped Newton iteration for ``residual(x) = 0``.

    Steps that leave the surface undefined (outside the domain or aperture)
    are halved until they land on valid parameters, so intersections near a
    patch boundary still converge instead of failing.

    Returns ``(x, f_norm, status)`` where ``status`` is one of
    ``_CONVERGED``, ``_STALLED``, or ``_EXHAUSTED``, or ``None`` if the
    iteration failed.
    """
    x = np.array(x0, dtype=float)
    status = _EXHAUSTED
    damped = 0
    F = residual(x)
    for _ in range(_c.MAX_NEWTON_ITER):
        if F is None or not np.all(np.isfinite(F)):
            return None
        f_norm = float(np.linalg.norm(F))
        if f_norm < _c.CONVERGED_F_NORM:
            return x, f_norm, _CONVERGED
        J = jacobian(x)
        if J is None or not np.all(np.isfinite(J)):
            return None
        delta = _newton_delta(J, F)
        if delta is None or not np.all(np.isfinite(delta)):
            return None
        if clamp:
            clamp_scale = max(1.0, float(np.max(np.abs(x))))
            if float(np.linalg.norm(delta)) > _c.CLAMP_REL * clamp_scale:
                delta = delta * (_c.CLAMP_REL * clamp_scale / float(np.linalg.norm(delta)))
        step = 1.0
        trial = x + delta
        Ft = residual(trial)
        while Ft is None or not np.all(np.isfinite(Ft)):
            step *= 0.5
            if step < _c.NEWTON_MIN_STEP:
                return None
            trial = x + step * delta
            Ft = residual(trial)
        if step < 1.0:
            damped += 1
        step_norm = float(np.linalg.norm(step * delta))
        x = trial
        F = Ft
        if step < _c.NEWTON_STALL_STEP or damped >= _c.NEWTON_MAX_DAMPED_STEPS:
            # Pinned against the edge of the domain: further iterations crawl.
            status = _STALLED
            break
        if step_norm < _c.STEP_REL_TOL * max(1.0, float(np.max(np.abs(x)))):
            status = _STALLED
            break
    if F is None or not np.all(np.isfinite(F)):
        return None
    return x, float(np.linalg.norm(F)), status


@njit(cache=True)
def _res_intersect(x: np.ndarray, desc: Any, o: np.ndarray, d: np.ndarray) -> np.ndarray:
    p = eval_njit(desc, float(x[1]), float(x[2]))
    w = desc[D_ROT] @ p + desc[D_TRANS]
    t = float(x[0])
    return w - (o + t * d)


@njit(cache=True)
def _jac_intersect(x: np.ndarray, desc: Any, o: np.ndarray, d: np.ndarray) -> np.ndarray:
    ru, rv = derivatives_njit(desc, float(x[1]), float(x[2]))
    R = desc[D_ROT]
    J = np.empty((3, 3))
    J[:, 0] = -d
    J[:, 1] = R @ ru
    J[:, 2] = R @ rv
    return J


@njit(cache=True)
def _res_xy(x: np.ndarray, desc: Any, x_target: float, y_target: float) -> np.ndarray:
    p = eval_njit(desc, float(x[0]), float(x[1]))
    return np.array([p[0] - x_target, p[1] - y_target])


@njit(cache=True)
def _jac_xy(x: np.ndarray, desc: Any, x_target: float, y_target: float) -> np.ndarray:
    ru, rv = derivatives_njit(desc, float(x[0]), float(x[1]))
    J = np.empty((2, 2))
    J[0, 0] = ru[0]
    J[0, 1] = rv[0]
    J[1, 0] = ru[1]
    J[1, 1] = rv[1]
    return J


@njit(cache=True)
def _residual_njit(
    kind: int, desc: Any, o: np.ndarray, d: np.ndarray, x_target: float, y_target: float, x: np.ndarray
) -> np.ndarray:
    """Newton residual for the selected problem (see ``_newton_njit``)."""
    if kind == _NEWTON_INTERSECT:
        return _res_intersect(x, desc, o, d)
    return _res_xy(x, desc, x_target, y_target)


@njit(cache=True)
def _jacobian_njit(
    kind: int, desc: Any, o: np.ndarray, d: np.ndarray, x_target: float, y_target: float, x: np.ndarray
) -> np.ndarray:
    """Newton Jacobian for the selected problem (see ``_newton_njit``)."""
    if kind == _NEWTON_INTERSECT:
        return _jac_intersect(x, desc, o, d)
    return _jac_xy(x, desc, x_target, y_target)


@njit(cache=True)
def _newton_njit(
    kind: int,
    desc: Any,
    o: np.ndarray,
    d: np.ndarray,
    x_target: float,
    y_target: float,
    x0: np.ndarray,
    clamp: bool,
) -> "tuple[np.ndarray, float, int] | None":
    """Damped Newton iteration in nopython mode; see ``_newton``.

    ``kind`` selects the residual/Jacobian problem (ray/surface
    intersection or local ``(x, y)`` inversion) by direct dispatch, so this
    stays a single compiled specialization instead of one per first-class
    function type.
    """
    x = x0.copy()
    status = _EXHAUSTED
    damped = 0
    F = _residual_njit(kind, desc, o, d, x_target, y_target, x)
    for _ in range(_c.MAX_NEWTON_ITER):
        if not np.all(np.isfinite(F)):
            return None
        f_norm = float(norm(F))
        if f_norm < _c.CONVERGED_F_NORM:
            return x, f_norm, _CONVERGED
        J = _jacobian_njit(kind, desc, o, d, x_target, y_target, x)
        if not np.all(np.isfinite(J)):
            return None
        delta = _newton_delta_njit(J, F)
        if delta is None:
            return None
        if not np.all(np.isfinite(delta)):
            return None
        if clamp:
            clamp_scale = max(1.0, float(np.max(np.abs(x))))
            if float(norm(delta)) > _c.CLAMP_REL * clamp_scale:
                delta = delta * (_c.CLAMP_REL * clamp_scale / float(norm(delta)))
        step = 1.0
        trial = x + delta
        Ft = _residual_njit(kind, desc, o, d, x_target, y_target, trial)
        while not np.all(np.isfinite(Ft)):
            step *= 0.5
            if step < _c.NEWTON_MIN_STEP:
                return None
            trial = x + step * delta
            Ft = _residual_njit(kind, desc, o, d, x_target, y_target, trial)
        if step < 1.0:
            damped += 1
        step_norm = float(norm(step * delta))
        x = trial
        F = Ft
        if step < _c.NEWTON_STALL_STEP or damped >= _c.NEWTON_MAX_DAMPED_STEPS:
            # Pinned against the edge of the domain: further iterations crawl.
            status = _STALLED
            break
        if step_norm < _c.STEP_REL_TOL * max(1.0, float(np.max(np.abs(x)))):
            status = _STALLED
            break
    if not np.all(np.isfinite(F)):
        return None
    return x, float(norm(F)), status
