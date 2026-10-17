"""Damped Newton iteration for the pure-Python reference oracle.

The production solver lives in the native kernels
(``optics._native.newton_intersection`` / ``solve_xy``); this Python twin is
the differential-testing oracle used by ``tests/_reference.py`` and keeps the
exact iteration semantics (damping, clamping, stall detection) that the C++
``newton_solve`` implements.
"""

from collections.abc import Callable, Sequence

import numpy as np

from .. import constants as _c
from .linalg import _newton_delta

_Residual = Callable[[np.ndarray], "np.ndarray | None"]
_Jacobian = Callable[[np.ndarray], "np.ndarray | None"]

# Solver outcome codes shared with the native kernels; they index the
# per-status tolerance tables in ``constants``.
_CONVERGED = _c.SOLVER_CONV
_STALLED = _c.SOLVER_STALL
_EXHAUSTED = _c.SOLVER_EXH


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
