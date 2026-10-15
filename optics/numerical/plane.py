"""Numba kernels shared between the measurement plane and the path tracer."""

import numpy as np
from numba import njit

from .linalg import _solve_small


@njit(cache=True)
def plane_intersect_kernel(
    ro: np.ndarray,
    d: np.ndarray,
    o: np.ndarray,
    n: np.ndarray,
    eps: float,
) -> "tuple[float, float, float, bool]":
    """Ray/plane intersection; returns ``(u, v, t, ok)`` in plane parameters.

    Solves the ray/parametrization equations for ``(u, v, t)`` with the
    plane taken as a graph over ``(u, v) = (x, y)``, dividing the plane
    equation by ``-n.z`` (guarded by the ``nz`` check above)::

        u = ro.x + t d.x,   v = ro.y + t d.y,
        (n.x/n.z)(u - o.x) + (n.y/n.z)(v - o.y) + (ro.z + t d.z - o.z) = 0
    """
    nz = n[2]
    if abs(nz) <= eps or abs(float(np.dot(d, n))) <= eps:
        return 0.0, 0.0, 0.0, False

    nxnz = n[0] / nz
    nynz = n[1] / nz
    A = np.array([
        [-1.0, 0.0, d[0]],
        [0.0, -1.0, d[1]],
        [nxnz, nynz, d[2]],
    ])
    b = np.array([-ro[0], -ro[1], nxnz * o[0] + nynz * o[1] + o[2] - ro[2]])

    det = d[2] + d[1] * nynz + d[0] * nxnz
    if not np.isfinite(det) or abs(det) <= eps:
        return 0.0, 0.0, 0.0, False

    sol = _solve_small(A, b)
    if sol is None:
        return 0.0, 0.0, 0.0, False
    u = float(sol[0])
    v = float(sol[1])
    t = float(sol[2])

    if not np.isfinite(t) or t <= 0.0:
        return 0.0, 0.0, 0.0, False
    if not (np.isfinite(u) and np.isfinite(v)):
        return 0.0, 0.0, 0.0, False
    return u, v, t, True
