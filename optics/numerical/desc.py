"""Surface-descriptor access kernels: evaluation, derivatives, membership."""

import math
from typing import Any

import numpy as np
from numba import njit

from .. import constants as _c
from ._kernels import (
    D_AP_CX,
    D_AP_CY,
    D_AP_R,
    D_DERIV,
    D_EVAL,
    D_TU,
    D_TV,
    D_U_MAX,
    D_U_MIN,
    D_V_MAX,
    D_V_MIN,
    build_descriptor,
)

# Descriptor and seed-grid tuple layouts (and their index constants) live in
# ``_kernels.py``; the callables in a descriptor are ``numba.cfunc`` pointers
# with a fixed C signature, so every surface shares one argument type: the
# kernels below compile (and disk-cache) once in total, not once per surface.


def build_surface_descriptor(surface: Any) -> "tuple[Any, ...]":
    """Flatten a ``Surface`` into the kernel descriptor layout.

    ``surface`` is duck-typed (cfunc pair, aperture, parameter range,
    transform) so this module keeps no import-time dependency on
    :mod:`optics.primitive`. Raises :class:`ValueError` when the surface
    carries an aperture the kernels cannot encode (see
    :meth:`~optics.primitive.aperture.Aperture.njit_params`); every compiled
    surface supports the nopython path, so there is no fallback.
    """
    if surface.aperture is None:
        ap_r, ap_cx, ap_cy = -1.0, 0.0, 0.0
    else:
        params = surface.aperture.njit_params()
        if params is None:
            raise ValueError(
                "Aperture type is not supported by the nopython surface kernels"
            )
        ap_r, ap_cx, ap_cy = params
    (u_min, u_max), (v_min, v_max) = surface.parameter_range
    tu = _c.DOMAIN_TOL * max(1.0, u_max - u_min)
    tv = _c.DOMAIN_TOL * max(1.0, v_max - v_min)
    return build_descriptor(
        surface._eval_cfn,
        surface._deriv_cfn,
        u_min,
        u_max,
        v_min,
        v_max,
        tu,
        tv,
        ap_r,
        ap_cx,
        ap_cy,
        surface.transform.rotation,
        surface.transform.translation,
    )


@njit(cache=True)
def eval_njit(desc: Any, u: float, v: float) -> np.ndarray:
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


@njit(cache=True)  # cfunc arg types are uniform, so one disk-cache entry fits all
def _probe_pair(eval_cfn: Any, deriv_cfn: Any) -> None:
    out = np.empty(3)
    ru = np.empty(3)
    rv = np.empty(3)
    eval_cfn(0.7, 0.35, out.ctypes.data)
    deriv_cfn(0.7, 0.35, ru.ctypes.data, rv.ctypes.data)


@njit(cache=True)
def _partial_njit(desc: Any, u: float, v: float, du: float, dv: float) -> np.ndarray:
    """One-sided finite-difference partial derivative (njit ``_partial_local``)."""
    base = eval_njit(desc, u, v)
    if not np.all(np.isfinite(base)):
        return np.full(3, math.nan)
    step = _c.FD_STEP_REL * max(1.0, abs(u), abs(v))
    for si in range(2):
        sign = 1.0 if si == 0 else -1.0
        h = step
        for _ in range(_c.FD_STEP_SHRINKS):
            p = eval_njit(desc, u + sign * du * h, v + sign * dv * h)
            if np.all(np.isfinite(p)):
                return (p - base) / (sign * h)
            h *= 0.5
    return np.full(3, math.nan)


@njit(cache=True)
def accepts_njit(desc: Any, u: float, v: float) -> bool:
    """Domain + circular-aperture membership (njit ``accepts_parameters``)."""
    if u < desc[D_U_MIN] - desc[D_TU] or u > desc[D_U_MAX] + desc[D_TU]:
        return False
    if v < desc[D_V_MIN] - desc[D_TV] or v > desc[D_V_MAX] + desc[D_TV]:
        return False
    if desc[D_AP_R] >= 0.0:
        p = eval_njit(desc, u, v)
        if not (math.isfinite(p[0]) and math.isfinite(p[1]) and math.isfinite(p[2])):
            return False
        dx = p[0] - desc[D_AP_CX]
        dy = p[1] - desc[D_AP_CY]
        return bool(dx * dx + dy * dy <= desc[D_AP_R] * desc[D_AP_R] + _c.APERTURE_EPS)
    return True


@njit(cache=True)
def derivatives_njit(desc: Any, u: float, v: float) -> "tuple[np.ndarray, np.ndarray]":
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
    frac = _c.DERIV_INTERIOR_FRAC * span / length
    for _ in range(_c.DERIV_INTERIOR_TRIES):
        if frac >= 1.0:
            break
        uu = u + du * frac
        vv = v + dv * frac
        if accepts_njit(desc, uu, vv):
            ru, rv = _deriv_pair_njit(desc, uu, vv)
            if np.all(np.isfinite(ru)) and np.all(np.isfinite(rv)):
                return ru, rv
        frac *= 2.0
    return (
        _partial_njit(desc, u, v, 1.0, 0.0),
        _partial_njit(desc, u, v, 0.0, 1.0),
    )
