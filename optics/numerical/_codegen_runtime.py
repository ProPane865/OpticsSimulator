"""Shared Numba kernels backing :mod:`optics.numerical.codegen`.

Everything a generated surface module used to JIT for *itself* lives here
instead: the safe div/mod helpers, the finite-difference tangent policy, the
cfunc plumbing, and the inert null-surface cfunc pair. These functions are
surface-independent, so they compile exactly once per process (and are
disk-cached across processes): the finite-difference kernels take the
surface's ``eval`` cfunc as a callback argument, and every surface's cfunc
shares the one fixed C signature, so Numba resolves them all to a single
specialization -- the same trick the descriptor kernels in ``desc.py`` use.

``optics.numerical.codegen`` calls :func:`warmup` at import so this one-time
compile happens at program loading, before any surface is built. The only
per-surface compilation left is the expression itself (``eval_local``,
``eval_grid``) and the two thin cfunc wrappers, which cannot be shared by
construction.
"""

import math
from typing import Any

import numpy as np
from numba import (  # type: ignore[attr-defined]  # carray re-exported via lazy module attr
    carray,
    cfunc,
    njit,
    types,
)
from numba.extending import intrinsic

from .. import constants as _c

# The cfuncs take raw output addresses (uint64) so every caller in the solver
# kernels sees one stable C signature; the int-to-pointer cast is confined to
# these cfunc bodies (which cannot be disk-cached anyway), keeping the nopython
# solver chain free of dynamic globals.
_UINT64 = types.uint64

EVAL_SIG = types.void(types.float64, types.float64, _UINT64)
DERIV_SIG = types.void(types.float64, types.float64, _UINT64, _UINT64)


@intrinsic
def _f64ptr_template(typingctx: Any, addr: Any) -> Any:
    ptr = types.CPointer(types.float64)
    sig = ptr(addr)

    def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
        llt = context.get_value_type(signature.return_type)
        return builder.inttoptr(args[0], llt)

    return sig, codegen


# ``_Intrinsic`` copies ``__wrapped__`` from the template definition, so mypy
# mistypes a call to it as a call of the (typingctx, addr) function. This
# ``Any`` alias is the real nopython-facing callable, and what generated
# modules import.
_int_to_f64ptr: Any = _f64ptr_template


@njit(cache=True)
def _sdiv(a: float, b: float) -> float:
    """Division with numpy scalar semantics (nan/inf instead of ZeroDivisionError)."""
    if b == 0.0:
        if a == 0.0:
            return math.nan
        return math.copysign(math.inf, a)
    return a / b


@njit(cache=True)
def _smod(a: float, b: float) -> float:
    """Modulo with numpy scalar semantics (nan instead of ZeroDivisionError)."""
    if b == 0.0:
        return math.nan
    return a % b


@njit(cache=True)
def _is_finite3(a: np.ndarray) -> bool:
    return math.isfinite(a[0]) and math.isfinite(a[1]) and math.isfinite(a[2])


@njit(cache=True)
def _eval_cb(eval_cb: Any, u: float, v: float) -> np.ndarray:
    out = np.empty(3)
    eval_cb(u, v, out.ctypes.data)
    return out


@njit(cache=True)
def _partial_rt(
    eval_cb: Any, u: float, v: float, h: float, lo: float, hi: float, along_u: bool
) -> np.ndarray:
    """One finite-difference partial along ``u`` (``along_u``) or ``v``.

    The policy is the Python mirror in optics.numerical.differentiation and
    samples the surface through its ``eval`` cfunc, so one compilation serves
    every surface.
    """
    if along_u:
        coord = u
        if lo <= u - h and u + h <= hi:
            f = _eval_cb(eval_cb, u + h, v)
            b = _eval_cb(eval_cb, u - h, v)
            if _is_finite3(f) and _is_finite3(b):
                return (f - b) / (2.0 * h)
    else:
        coord = v
        if lo <= v - h and v + h <= hi:
            f = _eval_cb(eval_cb, u, v + h)
            b = _eval_cb(eval_cb, u, v - h)
            if _is_finite3(f) and _is_finite3(b):
                return (f - b) / (2.0 * h)
    base = _eval_cb(eval_cb, u, v)
    if not _is_finite3(base):
        return np.full(3, math.nan)
    center = 0.5 * (lo + hi)
    inward = 1.0 if center >= coord else -1.0
    for s in range(2):
        sign = inward if s == 0 else -inward
        step = h
        for _ in range(_c.FD_STEP_SHRINKS):
            nxt = coord + sign * step
            if nxt < lo or nxt > hi:
                step *= 0.5
                continue
            if along_u:
                p = _eval_cb(eval_cb, nxt, v)
            else:
                p = _eval_cb(eval_cb, u, nxt)
            if _is_finite3(p):
                return (p - base) / (sign * step)
            step *= 0.5
    return np.full(3, math.nan)


@njit(cache=True)
def derivatives_rt(
    eval_cb: Any,
    u: float,
    v: float,
    hu: float,
    hv: float,
    umin: float,
    umax: float,
    vmin: float,
    vmax: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Tangents ``(r_u, r_v)`` for the surface behind ``eval_cb``."""
    ru = _partial_rt(eval_cb, u, v, hu, umin, umax, True)
    rv = _partial_rt(eval_cb, u, v, hv, vmin, vmax, False)
    return ru, rv


# --- Inert null-surface pair (``codegen.null_surface_cfuncs``) ---------------
# Padding trace contexts needs a cfunc pair with the surface signature that is
# never executed. It lives here as a static pair so building a padded context
# compiles nothing; the values mirror the ``("u", "v", "0")`` surface the
# generator used to emit for it.


@njit(cache=True)
def null_eval_local(u: float, v: float) -> np.ndarray:
    return np.array([u, v, 0.0], dtype=np.float64)


@njit(cache=True)
def null_derivatives_local(u: float, v: float) -> "tuple[np.ndarray, np.ndarray]":
    return (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )


@cfunc(EVAL_SIG)
def null_eval_cfn(u: float, v: float, addr: int) -> None:
    out = carray(_int_to_f64ptr(addr), 3)
    p = null_eval_local(u, v)
    out[0] = p[0]
    out[1] = p[1]
    out[2] = p[2]


@cfunc(DERIV_SIG)
def null_deriv_cfn(u: float, v: float, ru_addr: int, rv_addr: int) -> None:
    ru_out = carray(_int_to_f64ptr(ru_addr), 3)
    rv_out = carray(_int_to_f64ptr(rv_addr), 3)
    ru, rv = null_derivatives_local(u, v)
    ru_out[0] = ru[0]
    ru_out[1] = ru[1]
    ru_out[2] = ru[2]
    rv_out[0] = rv[0]
    rv_out[1] = rv[1]
    rv_out[2] = rv[2]


def warmup() -> None:
    """Force-compile every shared kernel and the null-surface cfunc pair.

    Called once when :mod:`optics.numerical.codegen` is imported, so surface
    creation never pays for the shared code path. Numba's disk cache makes
    later program starts cheap. Failures propagate: the chain is the same one
    every surface compiles against, so a failure is a real error.
    """
    from .desc import probe_njit_pair

    _sdiv(1.0, 2.0)
    _sdiv(1.0, 0.0)
    _smod(1.0, 2.0)
    _smod(1.0, 0.0)
    derivatives_rt(null_eval_cfn, 0.7, 0.35, 1e-3, 1e-3, 0.0, 1.0, 0.0, 1.0)
    if not probe_njit_pair(null_eval_cfn, null_deriv_cfn):
        raise RuntimeError("Could not compile null surface cfunc pair")
