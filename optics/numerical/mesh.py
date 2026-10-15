"""Numba kernels for ray/triangle-mesh intersection."""

import numpy as np
from numba import njit


@njit(cache=True)
def mesh_intersect_kernel(
    o: np.ndarray,
    d: np.ndarray,
    tri: np.ndarray,
    det_eps: float,
    bary_tol: float,
    min_face_t: float,
) -> "tuple[float, int, float, float, bool]":
    """Moller-Trumbore over ``tri`` (N, 3, 3); nearest forward hit.

    Returns ``(t, face, u, v, found)`` with barycentric ``u``/``v`` clamped
    into the triangle.
    """
    best_t = np.inf
    best_face = -1
    best_u = 0.0
    best_v = 0.0
    found = False

    for k in range(tri.shape[0]):
        a = tri[k, 0]
        e1 = tri[k, 1] - a
        e2 = tri[k, 2] - a

        h = np.cross(d, e2)
        det = float(np.dot(e1, h))
        if abs(det) <= det_eps:
            continue
        inv_det = 1.0 / det

        s = o - a
        u = float(np.dot(s, h)) * inv_det
        q = np.cross(s, e1)
        v = float(np.dot(q, d)) * inv_det
        t = float(np.dot(e2, q)) * inv_det

        if (
            t > min_face_t
            and u >= -bary_tol
            and v >= -bary_tol
            and u + v <= 1.0 + bary_tol
            and t < best_t
        ):
            best_t = t
            best_face = k
            best_u = u
            best_v = v
            found = True

    if not found:
        return 0.0, -1, 0.0, 0.0, False

    u_k = min(max(best_u, 0.0), 1.0)
    v_k = min(max(best_v, 0.0), 1.0 - u_k)
    return best_t, best_face, u_k, v_k, True
