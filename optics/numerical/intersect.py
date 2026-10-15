"""Nopython ray/surface intersection kernels."""

import math
from typing import Any

import numpy as np
from numba import njit

from .. import constants as _c
from ._kernels import G_P_WORLD, G_UF_VALID, G_VF_VALID
from .desc import accepts_njit
from .newton import _NEWTON_INTERSECT, _newton_njit


@njit(cache=True)
def _newton_intersection_njit(
    desc: Any,
    o: np.ndarray,
    d: np.ndarray,
    u: float,
    v: float,
    t: float,
) -> "tuple[float, float, float, bool]":
    """Full ray/surface Newton solve in nopython mode; returns ``(t, u, v, ok)``."""
    res = _newton_njit(
        _NEWTON_INTERSECT, desc, o, d, 0.0, 0.0, np.array([t, u, v]), True
    )
    if res is None:
        return 0.0, 0.0, 0.0, False
    x, f_norm, status = res
    if f_norm >= float(_c.HIT_TOLS[status]):
        return 0.0, 0.0, 0.0, False
    tt = float(x[0])
    uu = float(x[1])
    vv = float(x[2])
    if tt > 0.0 and accepts_njit(desc, uu, vv):
        return tt, uu, vv, True
    return 0.0, 0.0, 0.0, False


@njit(cache=True)
def surface_intersect_njit(
    desc: Any,
    grid: Any,
    o: np.ndarray,
    d: np.ndarray,
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
    tol = _c.SEED_TOL
    count = 0
    for i in range(m):
        if ts[i] > 0.0 and math.isfinite(residual[i]) and residual[i] <= tol:
            count += 1
    if count == 0:
        tol = _c.SEED_TOL_FALLBACK
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
    max_seeds = min(order.shape[0], _c.MAX_SEEDS)
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
        )
        if not ok:
            continue
        dup = False
        for h in range(n_hits):
            if abs(u - hit_u[h]) < _c.DEDUP_UV_TOL and abs(v - hit_v[h]) < _c.DEDUP_UV_TOL:
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
