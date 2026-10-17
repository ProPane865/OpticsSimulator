"""Native ray/triangle-mesh intersection kernel (Python entry point)."""

import numpy as np

from .. import _native


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
    return _native.mesh_intersect(
        np.ascontiguousarray(o, dtype=float),
        np.ascontiguousarray(d, dtype=float),
        np.ascontiguousarray(tri, dtype=float),
        float(det_eps),
        float(bary_tol),
        float(min_face_t),
    )
