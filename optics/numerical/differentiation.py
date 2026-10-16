"""Finite-difference derivative policy for parametric surfaces.

Symbolic derivatives are gone: tangents come from central differences whose step
scales with the parameter span, with an oriented one-sided fallback where a
central sample leaves the parameter domain or is non-finite (e.g. across a
``sqrt`` edge). The exact same policy is emitted into the generated Numba module
(see :mod:`optics.numerical.codegen`) and re-implemented here for the pure-Python
reference oracle, so the two agree point for point.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .. import constants as _c

_EVAL_FN = Callable[[float, float], np.ndarray]


def fd_step(lo: float, hi: float) -> float:
    """Range-aware finite-difference step, floored at ``FD_MIN_STEP``."""
    return max(_c.FD_STEP_REL * abs(hi - lo), _c.FD_MIN_STEP)


def _finite(point: np.ndarray) -> bool:
    return bool(np.isfinite(point).all())


def _partial(
    eval_fn: _EVAL_FN,
    u: float,
    v: float,
    axis: str,
    h: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """One tangent direction with central/oriented one-sided fallback."""

    def inside(x: float) -> bool:
        return lo <= x <= hi

    def sample(uu: float, vv: float) -> np.ndarray:
        return eval_fn(uu, vv)

    if axis == "u":
        lo_ok = inside(u - h)
        hi_ok = inside(u + h)
        forward = sample(u + h, v)
        backward = sample(u - h, v)
        base_coord = u
    else:
        lo_ok = inside(v - h)
        hi_ok = inside(v + h)
        forward = sample(u, v + h)
        backward = sample(u, v - h)
        base_coord = v

    # Central difference when both samples stay inside the parameter domain.
    if lo_ok and hi_ok and _finite(forward) and _finite(backward):
        return (forward - backward) / (2.0 * h)

    # Oriented one-sided differences, preferring the interior and shrinking.
    base = sample(u, v)
    if not _finite(base):
        return np.full(3, np.nan)
    center = 0.5 * (lo + hi)
    inward = 1.0 if center >= base_coord else -1.0
    for sign in (inward, -inward):
        step = h
        for _ in range(_c.FD_STEP_SHRINKS):
            nxt = base_coord + sign * step
            if not inside(nxt):
                step *= 0.5
                continue
            point = sample(nxt, v) if axis == "u" else sample(u, nxt)
            if _finite(point):
                return (point - base) / (sign * step)
            step *= 0.5
    return np.full(3, np.nan)


def numerical_derivatives(
    eval_fn: _EVAL_FN,
    u: float,
    v: float,
    u_range: Sequence[float],
    v_range: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(r_u, r_v)`` tangents for ``eval_fn`` at ``(u, v)``."""
    u_min, u_max = float(u_range[0]), float(u_range[1])
    v_min, v_max = float(v_range[0]), float(v_range[1])
    ru = _partial(eval_fn, u, v, "u", fd_step(u_min, u_max), u_min, u_max)
    rv = _partial(eval_fn, u, v, "v", fd_step(v_min, v_max), v_min, v_max)
    return ru, rv
