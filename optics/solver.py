import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from numba import njit

from . import constants as _c
from ._kernels import (
    D_AP_CX,
    D_AP_CY,
    D_AP_R,
    D_DERIV,
    D_EVAL,
    D_ROT,
    D_TRANS,
    D_TU,
    D_TV,
    D_U_MAX,
    D_U_MIN,
    D_V_MAX,
    D_V_MIN,
    G_FINITE,
    G_P_WORLD,
    G_PTS,
    G_UF,
    G_UF_VALID,
    G_VF,
    G_VF_VALID,
    build_grid,
)
from .hit import Hit
from .ray import Ray

if TYPE_CHECKING:
    from .surface import Surface

_MAX_NEWTON_ITER = _c.MAX_NEWTON_ITER
_STEP_REL_TOL = _c.STEP_REL_TOL
_CLAMP_REL = _c.CLAMP_REL
_CONVERGED_F_NORM = _c.CONVERGED_F_NORM
_DEDUP_UV_TOL = _c.DEDUP_UV_TOL
_MIN_STEP = _c.NEWTON_MIN_STEP
_STALL_STEP = _c.NEWTON_STALL_STEP
_MAX_DAMPED_STEPS = _c.NEWTON_MAX_DAMPED_STEPS

_CONVERGED = "converged"
_STALLED = "stalled"
_EXHAUSTED = "exhausted"

_PARAM_TOL = _c.PARAM_TOL
_HIT_TOL = _c.HIT_TOL

_Residual = Callable[[np.ndarray], "np.ndarray | None"]
_Jacobian = Callable[[np.ndarray], "np.ndarray | None"]

# --- Nopython solver path ---
# Status codes returned by the njit Newton loop (the Python path keeps the
# string statuses above; these feed the same per-status tolerance tables).
_S_CONV = 0
_S_STALL = 1
_S_EXH = 2

_PARAM_TOL_CONV = _c.PARAM_TOL[_CONVERGED]
_PARAM_TOL_STALL = _c.PARAM_TOL[_STALLED]
_PARAM_TOL_EXH = _c.PARAM_TOL[_EXHAUSTED]
_HIT_TOL_CONV = _c.HIT_TOL[_CONVERGED]
_HIT_TOL_STALL = _c.HIT_TOL[_STALLED]
_HIT_TOL_EXH = _c.HIT_TOL[_EXHAUSTED]

_SEED_TOL = _c.SEED_TOL
_SEED_TOL_FALLBACK = _c.SEED_TOL_FALLBACK
_MAX_SEEDS = _c.MAX_SEEDS
_XY_SEEDS = 12
_APERTURE_EPS = _c.APERTURE_EPS
_FD_STEP_REL = _c.FD_STEP_REL
_FD_STEP_SHRINKS = _c.FD_STEP_SHRINKS
_DERIV_INTERIOR_FRAC = _c.DERIV_INTERIOR_FRAC
_DERIV_INTERIOR_TRIES = _c.DERIV_INTERIOR_TRIES

# Descriptor and seed-grid tuple layouts (and their index constants) live in
# ``_kernels.py``; the callables in a descriptor are ``numba.cfunc`` pointers
# with a fixed C signature, so every surface shares one argument type: the
# kernels below compile (and disk-cache) once in total, not once per surface.


@njit(cache=True)
def _eval_njit(desc: Any, u: float, v: float) -> np.ndarray:
    out = np.empty(3)
    desc[D_EVAL](u, v, out.ctypes.data)
    return out


@njit(cache=True)
def _deriv_pair_njit(desc: Any, u: float, v: float) -> "tuple[np.ndarray, np.ndarray]":
    ru = np.empty(3)
    rv = np.empty(3)
    desc[D_DERIV](u, v, ru.ctypes.data, rv.ctypes.data)
    return ru, rv


def probe_njit_pair(eval_cfn: Any, deriv_cfn: Any) -> bool:
    """Force-compile the cfunc pair; False when they cannot be used."""
    try:
        _probe_pair(eval_cfn, deriv_cfn)
        return True
    except Exception:
        return False


@njit(cache=False)  # tiny; recompiled per process alongside the cfuncs
def _probe_pair(eval_cfn: Any, deriv_cfn: Any) -> None:
    out = np.empty(3)
    ru = np.empty(3)
    rv = np.empty(3)
    eval_cfn(0.7, 0.35, out.ctypes.data)
    deriv_cfn(0.7, 0.35, ru.ctypes.data, rv.ctypes.data)


class _SeedGrid(TypedDict):
    Uf: np.ndarray
    Vf: np.ndarray
    pts: np.ndarray
    finite: np.ndarray
    valid: np.ndarray
    Uf_valid: np.ndarray
    Vf_valid: np.ndarray
    P_world: np.ndarray


def _newton_delta(J: np.ndarray, F: np.ndarray) -> np.ndarray | None:
    """Solve ``J delta = -F``, falling back to least squares when rank-deficient."""
    try:
        if J.shape[0] == J.shape[1] and _is_nonsingular(J):
            return np.linalg.solve(J, -F)
        return np.linalg.lstsq(J, -F, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None


@njit(cache=True)
def _is_nonsingular(J: np.ndarray) -> bool:
    """Whether ``det(J)`` is large relative to the product of ``J``'s column norms."""
    rows, cols = J.shape
    if rows != cols:
        return False
    if rows == 3:
        a = J[0, 0]
        b = J[0, 1]
        c = J[0, 2]
        d = J[1, 0]
        e = J[1, 1]
        f = J[1, 2]
        g = J[2, 0]
        h = J[2, 1]
        i = J[2, 2]
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        scale = (
            math.sqrt(a * a + d * d + g * g)
            * math.sqrt(b * b + e * e + h * h)
            * math.sqrt(c * c + f * f + i * i)
        )
    elif rows == 2:
        a = J[0, 0]
        b = J[0, 1]
        c = J[1, 0]
        d = J[1, 1]
        det = a * d - b * c
        scale = math.sqrt(a * a + c * c) * math.sqrt(b * b + d * d)
    else:
        return False
    if scale <= 0.0 or not math.isfinite(scale):
        return False
    return bool(abs(det) / scale > _c.NEWTON_SINGULAR_REL)


def _newton(
    residual: _Residual,
    jacobian: _Jacobian,
    x0: Sequence[float],
    clamp: bool = False,
) -> tuple[np.ndarray, float, str] | None:
    """Damped Newton iteration for ``residual(x) = 0``.

    Steps that leave the surface undefined (outside the domain or aperture)
    are halved until they land on valid parameters, so intersections near a
    patch boundary still converge instead of failing.

    Returns ``(x, f_norm, status)`` where ``status`` is one of
    ``_CONVERGED``, ``_STALLED``, or ``_EXHAUSTED``, or
    ``(None, None, None)`` if the iteration failed.
    """
    x = np.array(x0, dtype=float)
    status = _EXHAUSTED
    damped = 0
    F = residual(x)
    for _ in range(_MAX_NEWTON_ITER):
        if F is None or not np.all(np.isfinite(F)):
            return None
        f_norm = float(np.linalg.norm(F))
        if f_norm < _CONVERGED_F_NORM:
            return x, f_norm, _CONVERGED
        J = jacobian(x)
        if J is None or not np.all(np.isfinite(J)):
            return None
        delta = _newton_delta(J, F)
        if delta is None or not np.all(np.isfinite(delta)):
            return None
        if clamp:
            clamp_scale = max(1.0, float(np.max(np.abs(x))))
            if float(np.linalg.norm(delta)) > _CLAMP_REL * clamp_scale:
                delta = delta * (_CLAMP_REL * clamp_scale / float(np.linalg.norm(delta)))
        step = 1.0
        trial = x + delta
        Ft = residual(trial)
        while Ft is None or not np.all(np.isfinite(Ft)):
            step *= 0.5
            if step < _MIN_STEP:
                return None
            trial = x + step * delta
            Ft = residual(trial)
        if step < 1.0:
            damped += 1
        step_norm = float(np.linalg.norm(step * delta))
        x = trial
        F = Ft
        if step < _STALL_STEP or damped >= _MAX_DAMPED_STEPS:
            # Pinned against the edge of the domain: further iterations crawl.
            status = _STALLED
            break
        if step_norm < _STEP_REL_TOL * max(1.0, float(np.max(np.abs(x)))):
            status = _STALLED
            break
    if F is None or not np.all(np.isfinite(F)):
        return None
    return x, float(np.linalg.norm(F)), status


@njit(cache=True)
def _norm(v: np.ndarray) -> float:
    """Euclidean norm.

    ``np.linalg.norm`` is avoided in nopython paths because its typing pass
    does not see through ``is None`` narrowing of optional arrays.
    """
    return float(math.sqrt(float(np.dot(v, v))))


@njit(cache=True)
def _solve_small(A: np.ndarray, b: np.ndarray) -> "np.ndarray | None":
    """Solve a 2x2/3x3 system by Gaussian elimination with partial pivoting."""
    n = A.shape[0]
    M = A.copy()
    y = b.copy()
    for k in range(n):
        p = k
        maxv = abs(M[k, k])
        for i in range(k + 1, n):
            if abs(M[i, k]) > maxv:
                maxv = abs(M[i, k])
                p = i
        if not math.isfinite(maxv) or maxv <= 0.0:
            return None
        if p != k:
            for j in range(n):
                tmp = M[k, j]
                M[k, j] = M[p, j]
                M[p, j] = tmp
            tmp = y[k]
            y[k] = y[p]
            y[p] = tmp
        for i in range(k + 1, n):
            f = M[i, k] / M[k, k]
            if f != 0.0:
                for j in range(k, n):
                    M[i, j] -= f * M[k, j]
                y[i] -= f * y[k]
    x = np.empty(n)
    for k in range(n - 1, -1, -1):
        s = y[k]
        for j in range(k + 1, n):
            s -= M[k, j] * x[j]
        x[k] = s / M[k, k]
    if not np.all(np.isfinite(x)):
        return None
    return x


@njit(cache=True)
def _newton_delta_njit(J: np.ndarray, F: np.ndarray) -> "np.ndarray | None":
    """Numpy-free ``_newton_delta``: square solve, else normal equations."""
    rows, cols = J.shape
    if rows == cols and (rows == 2 or rows == 3):
        if _is_nonsingular(J):
            return _solve_small(J, -F)
    if cols < 2 or cols > 3 or rows > 3:
        return None
    JtJ = J.T @ J
    if not _is_nonsingular(JtJ):
        return None
    return _solve_small(JtJ, J.T @ (-F))


_NEWTON_INTERSECT = 0
_NEWTON_XY = 1


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
    status = _S_EXH
    damped = 0
    if kind == _NEWTON_INTERSECT:
        F = _res_intersect(x, desc, o, d)
    else:
        F = _res_xy(x, desc, x_target, y_target)
    for _ in range(_MAX_NEWTON_ITER):
        if not np.all(np.isfinite(F)):
            return None
        f_norm = float(_norm(F))
        if f_norm < _CONVERGED_F_NORM:
            return x, f_norm, _S_CONV
        if kind == _NEWTON_INTERSECT:
            J = _jac_intersect(x, desc, o, d)
        else:
            J = _jac_xy(x, desc, x_target, y_target)
        if not np.all(np.isfinite(J)):
            return None
        delta = _newton_delta_njit(J, F)
        if delta is None:
            return None
        if not np.all(np.isfinite(delta)):
            return None
        if clamp:
            clamp_scale = max(1.0, float(np.max(np.abs(x))))
            if float(_norm(delta)) > _CLAMP_REL * clamp_scale:
                delta = delta * (_CLAMP_REL * clamp_scale / float(_norm(delta)))
        step = 1.0
        trial = x + delta
        if kind == _NEWTON_INTERSECT:
            Ft = _res_intersect(trial, desc, o, d)
        else:
            Ft = _res_xy(trial, desc, x_target, y_target)
        while not np.all(np.isfinite(Ft)):
            step *= 0.5
            if step < _MIN_STEP:
                return None
            trial = x + step * delta
            if kind == _NEWTON_INTERSECT:
                Ft = _res_intersect(trial, desc, o, d)
            else:
                Ft = _res_xy(trial, desc, x_target, y_target)
        if step < 1.0:
            damped += 1
        step_norm = float(_norm(step * delta))
        x = trial
        F = Ft
        if step < _STALL_STEP or damped >= _MAX_DAMPED_STEPS:
            status = _S_STALL
            break
        if step_norm < _STEP_REL_TOL * max(1.0, float(np.max(np.abs(x)))):
            status = _S_STALL
            break
    if not np.all(np.isfinite(F)):
        return None
    return x, float(_norm(F)), status


@njit(cache=True)
def _partial_njit(desc: Any, u: float, v: float, du: float, dv: float) -> np.ndarray:
    """One-sided finite-difference partial derivative (njit ``_partial_local``)."""
    base = _eval_njit(desc, u, v)
    if not np.all(np.isfinite(base)):
        return np.full(3, math.nan)
    step = _FD_STEP_REL * max(1.0, abs(u), abs(v))
    for si in range(2):
        sign = 1.0 if si == 0 else -1.0
        h = step
        for _ in range(_FD_STEP_SHRINKS):
            p = _eval_njit(desc, u + sign * du * h, v + sign * dv * h)
            if np.all(np.isfinite(p)):
                return (p - base) / (sign * h)
            h *= 0.5
    return np.full(3, math.nan)


@njit(cache=True)
def _accepts_njit(desc: Any, u: float, v: float) -> bool:
    """Domain + circular-aperture membership (njit ``accepts_parameters``)."""
    if u < desc[D_U_MIN] - desc[D_TU] or u > desc[D_U_MAX] + desc[D_TU]:
        return False
    if v < desc[D_V_MIN] - desc[D_TV] or v > desc[D_V_MAX] + desc[D_TV]:
        return False
    if desc[D_AP_R] >= 0.0:
        p = _eval_njit(desc, u, v)
        if not (math.isfinite(p[0]) and math.isfinite(p[1]) and math.isfinite(p[2])):
            return False
        dx = p[0] - desc[D_AP_CX]
        dy = p[1] - desc[D_AP_CY]
        return bool(dx * dx + dy * dy <= desc[D_AP_R] * desc[D_AP_R] + _APERTURE_EPS)
    return True


@njit(cache=True)
def _derivatives_njit(desc: Any, u: float, v: float) -> "tuple[np.ndarray, np.ndarray]":
    """Analytic tangents with the interior pull-back / FD fallback in nopython."""
    ru, rv = _deriv_pair_njit(desc, u, v)
    if np.all(np.isfinite(ru)) and np.all(np.isfinite(rv)):
        return ru, rv
    u_min = desc[D_U_MIN]
    u_max = desc[D_U_MAX]
    v_min = desc[D_V_MIN]
    v_max = desc[D_V_MAX]
    du = 0.5 * (u_min + u_max) - u
    dv = 0.5 * (v_min + v_max) - v
    if du == 0.0 and dv == 0.0:
        return (
            _partial_njit(desc, u, v, 1.0, 0.0),
            _partial_njit(desc, u, v, 0.0, 1.0),
        )
    span = max(abs(u_max - u_min), abs(v_max - v_min), 1.0)
    length = math.hypot(du, dv)
    frac = _DERIV_INTERIOR_FRAC * span / length
    for _ in range(_DERIV_INTERIOR_TRIES):
        if frac >= 1.0:
            break
        uu = u + du * frac
        vv = v + dv * frac
        if _accepts_njit(desc, uu, vv):
            ru, rv = _deriv_pair_njit(desc, uu, vv)
            if np.all(np.isfinite(ru)) and np.all(np.isfinite(rv)):
                return ru, rv
        frac *= 2.0
    return (
        _partial_njit(desc, u, v, 1.0, 0.0),
        _partial_njit(desc, u, v, 0.0, 1.0),
    )


@njit(cache=True)
def _res_intersect(x: np.ndarray, desc: Any, o: np.ndarray, d: np.ndarray) -> np.ndarray:
    p = _eval_njit(desc, float(x[1]), float(x[2]))
    w = desc[D_ROT] @ p + desc[D_TRANS]
    t = float(x[0])
    return w - (o + t * d)


@njit(cache=True)
def _jac_intersect(x: np.ndarray, desc: Any, o: np.ndarray, d: np.ndarray) -> np.ndarray:
    ru, rv = _derivatives_njit(desc, float(x[1]), float(x[2]))
    R = desc[D_ROT]
    J = np.empty((3, 3))
    J[:, 0] = -d
    J[:, 1] = R @ ru
    J[:, 2] = R @ rv
    return J


@njit(cache=True)
def _newton_intersection_njit(
    desc: Any,
    o: np.ndarray,
    d: np.ndarray,
    u: float,
    v: float,
    t: float,
    tol_conv: float,
    tol_stall: float,
    tol_exh: float,
) -> "tuple[float, float, float, bool]":
    """Full ray/surface Newton solve in nopython mode; returns ``(t, u, v, ok)``."""
    res = _newton_njit(
        _NEWTON_INTERSECT, desc, o, d, 0.0, 0.0, np.array([t, u, v]), True
    )
    if res is None:
        return 0.0, 0.0, 0.0, False
    x, f_norm, status = res
    if status == _S_CONV:
        tol = tol_conv
    elif status == _S_STALL:
        tol = tol_stall
    else:
        tol = tol_exh
    if f_norm >= tol:
        return 0.0, 0.0, 0.0, False
    tt = float(x[0])
    uu = float(x[1])
    vv = float(x[2])
    if tt > 0.0 and _accepts_njit(desc, uu, vv):
        return tt, uu, vv, True
    return 0.0, 0.0, 0.0, False


@njit(cache=True)
def _res_xy(x: np.ndarray, desc: Any, x_target: float, y_target: float) -> np.ndarray:
    p = _eval_njit(desc, float(x[0]), float(x[1]))
    return np.array([p[0] - x_target, p[1] - y_target])


@njit(cache=True)
def _jac_xy(x: np.ndarray, desc: Any, x_target: float, y_target: float) -> np.ndarray:
    ru, rv = _derivatives_njit(desc, float(x[0]), float(x[1]))
    J = np.empty((2, 2))
    J[0, 0] = ru[0]
    J[0, 1] = rv[0]
    J[1, 0] = ru[1]
    J[1, 1] = rv[1]
    return J


@njit(cache=True)
def _solve_xy_njit(
    desc: Any, x_target: float, y_target: float, u: float, v: float
) -> "tuple[float, float, bool]":
    """Newton inversion of local ``(x, y)`` to parameters, in nopython mode."""
    res = _newton_njit(
        _NEWTON_XY,
        desc,
        np.zeros(3),
        np.zeros(3),
        x_target,
        y_target,
        np.array([u, v]),
        False,
    )
    if res is None:
        return 0.0, 0.0, False
    x, f_norm, status = res
    if status == _S_CONV:
        tol = _PARAM_TOL_CONV
    elif status == _S_STALL:
        tol = _PARAM_TOL_STALL
    else:
        tol = _PARAM_TOL_EXH
    if f_norm >= tol:
        return 0.0, 0.0, False
    u = float(x[0])
    v = float(x[1])
    if not _accepts_njit(desc, u, v):
        return 0.0, 0.0, False
    return u, v, True


@njit(cache=True)
def _parameters_at_xy_njit(
    desc: Any, grid: Any, x: float, y: float
) -> "tuple[float, float, bool]":
    """Grid-seeded local ``(x, y)`` inversion in nopython mode."""
    pts = grid[G_PTS]
    finite = grid[G_FINITE]
    Uf = grid[G_UF]
    Vf = grid[G_VF]
    n = pts.shape[0]
    if n == 0:
        return 0.0, 0.0, False
    d2 = np.empty(n)
    any_fin = False
    for i in range(n):
        if finite[i]:
            dx = pts[i, 0] - x
            dy = pts[i, 1] - y
            d2[i] = dx * dx + dy * dy
            any_fin = True
        else:
            d2[i] = np.inf
    if not any_fin:
        return 0.0, 0.0, False
    k = min(_XY_SEEDS, n)
    order = np.argpartition(d2, k - 1)[:k]
    order = order[np.argsort(d2[order])]
    for oi in range(order.shape[0]):
        idx = int(order[oi])
        if not finite[idx]:
            continue
        u, v, ok = _solve_xy_njit(desc, x, y, float(Uf[idx]), float(Vf[idx]))
        if ok:
            return u, v, True
    return 0.0, 0.0, False


@njit(cache=True)
def _surface_intersect_njit(
    desc: Any,
    grid: Any,
    o: np.ndarray,
    d: np.ndarray,
    tol_conv: float,
    tol_stall: float,
    tol_exh: float,
) -> "tuple[float, float, float, bool]":
    """Seed selection + Newton + dedup, the whole ray/surface intersection.

    Nopython counterpart of ``SurfaceSolver.intersect``; returns the nearest
    forward hit as ``(t, u, v, ok)``.
    """
    P = grid[G_P_WORLD]
    Uf = grid[G_UF_VALID]
    Vf = grid[G_VF_VALID]
    m = P.shape[0]
    if m == 0:
        return 0.0, 0.0, 0.0, False
    ts = np.empty(m)
    residual = np.empty(m)
    for i in range(m):
        rel0 = P[i, 0] - o[0]
        rel1 = P[i, 1] - o[1]
        rel2 = P[i, 2] - o[2]
        t = rel0 * d[0] + rel1 * d[1] + rel2 * d[2]
        ts[i] = t
        if t > 0.0:
            px = o[0] + t * d[0] - P[i, 0]
            py = o[1] + t * d[1] - P[i, 1]
            pz = o[2] + t * d[2] - P[i, 2]
            residual[i] = math.sqrt(px * px + py * py + pz * pz)
        else:
            residual[i] = np.inf
    tol = _SEED_TOL
    count = 0
    for i in range(m):
        if ts[i] > 0.0 and math.isfinite(residual[i]) and residual[i] <= tol:
            count += 1
    if count == 0:
        tol = _SEED_TOL_FALLBACK
        count = 0
        for i in range(m):
            if ts[i] > 0.0 and math.isfinite(residual[i]) and residual[i] <= tol:
                count += 1
        if count == 0:
            return 0.0, 0.0, 0.0, False
    idx = np.empty(count, dtype=np.int64)
    c = 0
    for i in range(m):
        if ts[i] > 0.0 and math.isfinite(residual[i]) and residual[i] <= tol:
            idx[c] = i
            c += 1
    order = idx[np.argsort(residual[idx])]
    max_seeds = min(order.shape[0], _MAX_SEEDS)
    hit_t = np.empty(max_seeds)
    hit_u = np.empty(max_seeds)
    hit_v = np.empty(max_seeds)
    n_hits = 0
    for k in range(max_seeds):
        i = int(order[k])
        t, u, v, ok = _newton_intersection_njit(
            desc,
            o,
            d,
            float(Uf[i]),
            float(Vf[i]),
            float(ts[i]),
            tol_conv,
            tol_stall,
            tol_exh,
        )
        if not ok:
            continue
        dup = False
        for h in range(n_hits):
            if abs(u - hit_u[h]) < _DEDUP_UV_TOL and abs(v - hit_v[h]) < _DEDUP_UV_TOL:
                if t < hit_t[h]:
                    hit_t[h] = t
                    hit_u[h] = u
                    hit_v[h] = v
                dup = True
                break
        if not dup:
            hit_t[n_hits] = t
            hit_u[n_hits] = u
            hit_v[n_hits] = v
            n_hits += 1
    if n_hits == 0:
        return 0.0, 0.0, 0.0, False
    best = 0
    for h in range(1, n_hits):
        if hit_t[h] < hit_t[best]:
            best = h
    return hit_t[best], hit_u[best], hit_v[best], True


class SurfaceSolver:
    surface: "Surface"

    def __init__(self, surface: "Surface") -> None:
        self.surface = surface
        self._grid_cache: dict[int, _SeedGrid] = {}
        self._njit_grid_cache: dict[int, tuple[Any, ...]] = {}

    def njit_grid(self, grid_n: int = _c.SEED_GRID_N) -> tuple[Any, ...]:
        """Seed grid as a flat tuple of contiguous arrays for the njit kernels."""
        cached = self._njit_grid_cache.get(grid_n)
        if cached is None:
            g = self._seed_grid(grid_n)
            cached = build_grid(
                g["P_world"],
                g["Uf_valid"],
                g["Vf_valid"],
                g["pts"],
                g["finite"],
                g["Uf"],
                g["Vf"],
            )
            self._njit_grid_cache[grid_n] = cached
        return cached

    def parameters_at(self, pos: np.ndarray | Sequence[float]) -> "tuple[float, float] | None":
        p = self.surface.transform.point_to_local(np.asarray(pos, dtype=float))
        if not np.all(np.isfinite(p)):
            return None
        u0 = float(p[0])
        v0 = float(p[1])
        if self.surface.accepts_parameters(u0, v0):
            puv = self.surface.evaluate_local(u0, v0)
            if np.all(np.isfinite(puv)) and float(np.linalg.norm(puv - p)) < _c.POINT_MATCH_TOL:
                return u0, v0
        grid = self._seed_grid()
        pts = grid["pts"]
        dist = np.linalg.norm(pts - p, axis=1)
        valid = grid["valid"] & np.isfinite(dist)
        if not valid.any():
            return None
        idx = int(np.argmin(np.where(valid, dist, np.inf)))
        u = float(grid["Uf"][idx])
        v = float(grid["Vf"][idx])
        return self._solve_parameters_point(p, u, v)

    def parameters_at_xy(
        self,
        x: float,
        y: float,
        seed: "tuple[float, float] | None" = None,
    ) -> "tuple[float, float] | None":
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

        distance2 = np.where(finite, ((pts[:, 0] - x)**2 + (pts[:, 1] - y)**2), np.inf)
        k = min(12, distance2.size)
        order = np.argpartition(distance2, k - 1)[:k]
        order = order[np.argsort(distance2[order])]

        Uf = grid["Uf"]
        Vf = grid["Vf"]

        for idx in order:
            result = self._solve_parameters_xy(x, y, float(Uf[idx]), float(Vf[idx]))

            if result is not None:
                return result

        return None

    def intersect(self, ray: Ray) -> Hit | None:
        o = np.asarray(ray.origin, dtype=float)
        d = np.asarray(ray.direction, dtype=float)
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
            return None
        ray = Ray(o, d, ray.power)
        hits: list[Hit] = []
        for u0, v0, t0 in self._intersection_seeds(ray):
            hit = self._newton_intersection(ray, u0, v0, t0)
            if hit is None:
                continue
            for i, other in enumerate(hits):
                if (
                    abs(hit.u - other.u) < _DEDUP_UV_TOL
                    and abs(hit.v - other.v) < _DEDUP_UV_TOL
                ):
                    if hit.t < other.t:
                        hits[i] = hit
                    break
            else:
                hits.append(hit)
        if not hits:
            return None
        return min(hits, key=lambda h: h.t)

    def _solve_parameters_point(
        self, p: np.ndarray, u: float, v: float
    ) -> "tuple[float, float] | None":
        def residual(x: np.ndarray) -> np.ndarray:
            return self.surface.evaluate_local(float(x[0]), float(x[1])) - p

        def jacobian(x: np.ndarray) -> np.ndarray:
            ru, rv = self.surface.derivatives_local(float(x[0]), float(x[1]))
            return np.column_stack((ru, rv))

        result = _newton(residual, jacobian, (u, v))
        if result is None:
            return None
        x, f_norm, status = result
        if f_norm >= _PARAM_TOL[status]:
            return None
        u = float(x[0])
        v = float(x[1])
        if not self.surface.accepts_parameters(u, v):
            return None
        return u, v

    def _solve_parameters_xy(
        self, x_target: float, y_target: float, u: float, v: float
    ) -> "tuple[float, float] | None":
        desc = self.surface.njit_descriptor
        if desc is not None:
            uu, vv, ok = _solve_xy_njit(desc, x_target, y_target, u, v)
            if not ok:
                return None
            return uu, vv
        return self._solve_parameters_xy_python(x_target, y_target, u, v)

    def _solve_parameters_xy_python(
        self, x_target: float, y_target: float, u: float, v: float
    ) -> "tuple[float, float] | None":
        def residual(x: np.ndarray) -> np.ndarray:
            p = self.surface.evaluate_local(float(x[0]), float(x[1]))
            return np.array([p[0] - x_target, p[1] - y_target])

        def jacobian(x: np.ndarray) -> np.ndarray:
            ru, rv = self.surface.derivatives_local(float(x[0]), float(x[1]))
            return np.array([
                [ru[0], rv[0]],
                [ru[1], rv[1]],
            ])

        result = _newton(residual, jacobian, (u, v))
        if result is None:
            return None
        x, f_norm, status = result
        if f_norm >= _PARAM_TOL[status]:
            return None
        u = float(x[0])
        v = float(x[1])
        if not self.surface.accepts_parameters(u, v):
            return None
        return u, v

    def _aperture_mask(self, uf: np.ndarray, vf: np.ndarray, pts: np.ndarray) -> np.ndarray:
        s = self.surface
        assert s.aperture is not None
        return np.asarray(
            s.aperture.mask(s, uf, vf, pts[:, 0], pts[:, 1], pts[:, 2]),
            dtype=bool,
        )

    def _grid_mask(self, U: np.ndarray, V: np.ndarray, pts: np.ndarray) -> np.ndarray:
        finite = np.all(np.isfinite(pts), axis=1)
        uf = U.ravel()
        vf = V.ravel()
        mask = finite & self.surface.in_domain(uf, vf)
        if self.surface.aperture is not None:
            mask = mask & self._aperture_mask(uf, vf, pts)
        return mask

    def _seed_grid(self, grid_n: int = _c.SEED_GRID_N) -> _SeedGrid:
        entry = self._grid_cache.get(grid_n)
        if entry is not None:
            return entry
        s = self.surface
        (u_min, u_max), (v_min, v_max) = s.parameter_range
        us = np.linspace(u_min, u_max, grid_n)
        vs = np.linspace(v_min, v_max, grid_n)
        U, V = np.meshgrid(us, vs)
        Uf = U.ravel()
        Vf = V.ravel()
        pts = s.evaluate_local(U, V).reshape(-1, 3)
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
                s.transform.points_to_world(pts[valid])
                if valid.any()
                else np.empty((0, 3), dtype=float)
            ),
        }
        self._grid_cache[grid_n] = entry
        return entry

    def _intersection_seeds(
        self,
        ray: Ray,
        grid_n: int = _c.SEED_GRID_N,
        seed_tol: float = _c.SEED_TOL,
        max_seeds: int = _c.MAX_SEEDS,
    ) -> list[tuple[float, float, float]]:
        grid = self._seed_grid(grid_n)
        if not grid["valid"].any():
            return []
        P = grid["P_world"]
        rel = P - ray.origin
        t = rel @ ray.direction
        forward = t > 0.0
        if not forward.any():
            return []
        proj = ray.at(t[:, None])
        residual = np.linalg.norm(P - proj, axis=1)
        mask = forward & np.isfinite(residual) & (residual <= seed_tol)
        if not mask.any():
            mask = forward & np.isfinite(residual) & (residual <= _c.SEED_TOL_FALLBACK)
            if not mask.any():
                return []
        indices = np.where(mask)[0]
        order = indices[np.argsort(residual[indices])]
        Uf = grid["Uf_valid"]
        Vf = grid["Vf_valid"]
        return [
            (float(Uf[idx]), float(Vf[idx]), float(t[idx]))
            for idx in order[:max_seeds]
        ]

    def _accept_intersection(self, ray: Ray, t: float, u: float, v: float) -> Hit | None:
        if t > 0.0 and self.surface.accepts_parameters(u, v):
            return Hit(ray.at(t), float(t), float(u), float(v))
        return None

    def _newton_intersection(self, ray: Ray, u: float, v: float, t: float) -> Hit | None:
        desc = self.surface.njit_descriptor
        if desc is not None:
            tt, uu, vv, ok = _newton_intersection_njit(
                desc,
                np.asarray(ray.origin, dtype=float),
                np.asarray(ray.direction, dtype=float),
                u,
                v,
                t,
                _HIT_TOL_CONV,
                _HIT_TOL_STALL,
                _HIT_TOL_EXH,
            )
            if not ok:
                return None
            return Hit(ray.at(tt), tt, uu, vv)
        return self._newton_intersection_python(ray, u, v, t)

    def _newton_intersection_python(
        self, ray: Ray, u: float, v: float, t: float
    ) -> Hit | None:
        def residual(x: np.ndarray) -> np.ndarray:
            return self.surface.point(float(x[1]), float(x[2])) - ray.at(float(x[0]))

        def jacobian(x: np.ndarray) -> np.ndarray:
            ru, rv = self.surface.tangents(float(x[1]), float(x[2]))
            return np.column_stack((-ray.direction, ru, rv))

        result = _newton(residual, jacobian, (t, u, v), clamp=True)
        if result is None:
            return None
        x, f_norm, status = result
        if f_norm >= _HIT_TOL[status]:
            return None
        return self._accept_intersection(ray, float(x[0]), float(x[1]), float(x[2]))
