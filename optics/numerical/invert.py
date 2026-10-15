"""Nopython kernels inverting local (x, y) targets to surface parameters."""

from typing import Any

import numpy as np
from numba import njit

from .. import constants as _c
from ._kernels import G_FINITE, G_PTS, G_UF, G_VF
from .desc import accepts_njit
from .newton import _NEWTON_XY, _newton_njit


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
    if f_norm >= float(_c.PARAM_TOLS[status]):
        return 0.0, 0.0, False
    u = float(x[0])
    v = float(x[1])
    if not accepts_njit(desc, u, v):
        return 0.0, 0.0, False
    return u, v, True


@njit(cache=True)
def parameters_at_xy_njit(
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
    k = min(_c.XY_SEEDS, n)
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
