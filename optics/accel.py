"""Numba-accelerated ray tracing kernels.

Numba is a required dependency; this module is the single ray-tracing backend.
Surface geometry is re-implemented here in a Numba-friendly form from each
surface's ``numba_descriptor`` (``(kind, coeffs_1d)``).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

HAVE_NUMBA = True

from .surfaces import CONIC, PLANE


@njit(cache=True)
def _newton(kind, c0, o, d, max_iter=40, tol=1e-12):
    x0, y0, z0 = o[0], o[1], o[2]
    dx, dy, dz = d[0], d[1], d[2]

    if kind == PLANE:
        if abs(dz) < 1e-12:
            return 1e-9
        return (c0[0] - z0) / dz

    if kind == CONIC:
        c, k, zc = c0[0], c0[1], c0[2]
        if abs(dz) > 1e-12:
            t = (zc - z0) / dz
        else:
            t = 1e-3
        for _ in range(max_iter):
            px, py, pz = x0 + t * dx, y0 + t * dy, z0 + t * dz
            r2 = px * px + py * py
            s = 1.0 - (1.0 - k) * c * c * r2
            if s < 1e-15:
                s = 1e-15
            sq = math.sqrt(s)
            D = 1.0 + sq
            z = c * r2 / D + zc
            f = (c / (D * D)) * (2.0 * D + (1.0 - k) * c * c * r2 / sq)
            gx, gy = f * px, f * py
            func = z - (z0 + t * dz)
            dfdt = gx * dx + gy * dy - dz
            if abs(dfdt) < 1e-30:
                break
            step = func / dfdt
            if step != step:
                break
            tn = t - step
            if tn < 1e-12:
                tn = 1e-12
            if abs(tn - t) < tol:
                t = tn
                break
            t = tn
        return t

    order = int(c0[0])
    zc = c0[1]
    ncoef = (order + 1) * (order + 1)
    C = np.empty((order + 1, order + 1))
    for i in range(ncoef):
        C[i // (order + 1), i % (order + 1)] = c0[2 + i]
    if abs(dz) > 1e-12:
        t = (zc - z0) / dz
    else:
        t = 1e-3
    for _ in range(max_iter):
        px, py, pz = x0 + t * dx, y0 + t * dy, z0 + t * dz
        z = zc
        gx = 0.0
        gy = 0.0
        xa = 1.0
        xa_prev = 1.0
        ya = 1.0
        ya_prev = 1.0
        for a in range(order + 1):
            if a > 0:
                xa_prev = xa
                xa = xa * px
            ya = 1.0
            ya_prev = 1.0
            for b in range(order + 1):
                term = C[a, b] * xa * ya
                z = z + term
                if a > 0:
                    gx = gx + a * C[a, b] * xa_prev * ya
                if b > 0:
                    gy = gy + b * C[a, b] * xa * ya_prev
                if b < order:
                    ya_prev = ya
                    ya = ya * py
        func = z - (z0 + t * dz)
        dfdt = gx * dx + gy * dy - dz
        if abs(dfdt) < 1e-30:
            break
        step = func / dfdt
        if step != step:
            break
        tn = t - step
        if tn < 1e-12:
            tn = 1e-12
        if abs(tn - t) < tol:
            t = tn
            break
        t = tn
    return t


@njit(cache=True)
def _normal(kind, c0, p):
    if kind == PLANE:
        return np.array([0.0, 0.0, 1.0])
    px, py = p[0], p[1]
    if kind == CONIC:
        c, k = c0[0], c0[1]
        r2 = px * px + py * py
        s = 1.0 - (1.0 - k) * c * c * r2
        if s < 1e-15:
            s = 1e-15
        sq = math.sqrt(s)
        D = 1.0 + sq
        f = (c / (D * D)) * (2.0 * D + (1.0 - k) * c * c * r2 / sq)
        gx, gy = f * px, f * py
    else:
        order = int(c0[0])
        ncoef = (order + 1) * (order + 1)
        C = np.empty((order + 1, order + 1))
        for i in range(ncoef):
            C[i // (order + 1), i % (order + 1)] = c0[2 + i]
        gx = 0.0
        gy = 0.0
        xa = 1.0
        xa_prev = 1.0
        ya = 1.0
        ya_prev = 1.0
        for a in range(order + 1):
            if a > 0:
                xa_prev = xa
                xa = xa * px
            ya = 1.0
            ya_prev = 1.0
            for b in range(order + 1):
                if a > 0:
                    gx = gx + a * C[a, b] * xa_prev * ya
                if b > 0:
                    gy = gy + b * C[a, b] * xa * ya_prev
                if b < order:
                    ya_prev = ya
                    ya = ya * py
    length = math.sqrt(gx * gx + gy * gy + 1.0)
    return np.array([-gx / length, -gy / length, 1.0 / length])


@njit(cache=True)
def _aperture(kind, c0):
    if kind == PLANE:
        return c0[1]
    if kind == CONIC:
        return c0[3]
    return -1.0


@njit(parallel=True, cache=True)
def _trace_kernel(origins, dirs, kinds, flat_coeffs, offsets, media):
    S = kinds.shape[0]
    N = origins.shape[0]
    hits = np.empty((S, N, 3))
    tir_all = np.zeros((S, N), np.int8)
    blocked = np.zeros(N, np.int8)

    for r in prange(N):
        d = dirs[r]
        norm = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        if norm > 0:
            d = d / norm
        o = origins[r].copy()

        for s in range(S):
            if blocked[r]:
                break
            kind = kinds[s]
            c0 = flat_coeffs[offsets[s]:offsets[s + 1]]
            t = _newton(kind, c0, o, d)
            hit = o + t * d
            hits[s, r] = hit

            ap2 = _aperture(kind, c0)
            if ap2 >= 0.0 and (hit[0] * hit[0] + hit[1] * hit[1]) > ap2:
                blocked[r] = 1
                continue

            nn = _normal(kind, c0, hit)
            cos_i = -(nn[0] * d[0] + nn[1] * d[1] + nn[2] * d[2])
            if cos_i < 0.0:
                nn = -nn
                cos_i = -cos_i

            eta = media[s] / media[s + 1]
            sin2 = eta * eta * (1.0 - cos_i * cos_i)
            tir = sin2 > 1.0
            if tir:
                tir_all[s, r] = 1
            if sin2 > 1.0:
                sin2 = 1.0
            cost = math.sqrt(max(1.0 - sin2, 0.0))

            dr = eta * d + (eta * cos_i - cost) * nn
            dotn = d[0] * nn[0] + d[1] * nn[1] + d[2] * nn[2]
            refl = d - 2.0 * dotn * nn
            d = np.where(tir, refl, dr)

            dnorm = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            if dnorm > 0:
                d = d / dnorm

            o = hit

        origins[r] = o
        dirs[r] = d

    return hits, tir_all, blocked


def trace_numba(origins, dirs, surfaces, media):
    """Accelerated trace. Returns ``(hits (S,N,3), tir (S,N), blocked (N,))``."""
    origins = np.ascontiguousarray(origins, dtype=np.float64)
    dirs = np.ascontiguousarray(dirs, dtype=np.float64)
    kinds = np.array([s.kind for s in surfaces], dtype=np.int64)
    descs = [s.numba_descriptor() for s in surfaces]
    flat_coeffs = np.concatenate([c for _, c in descs])
    offsets = np.zeros(len(surfaces) + 1, dtype=np.int64)
    for i, (_, c) in enumerate(descs):
        offsets[i + 1] = offsets[i] + len(c)
    media_arr = np.asarray(media, dtype=np.float64)
    hits, tir, blocked = _trace_kernel(
        origins, dirs, kinds, flat_coeffs, offsets, media_arr)
    return (np.ascontiguousarray(hits),
            np.ascontiguousarray(tir).astype(bool),
            np.ascontiguousarray(blocked).astype(bool))
