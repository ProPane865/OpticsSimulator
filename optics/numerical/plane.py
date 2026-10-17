"""Native ray/plane kernel (Python entry point)."""

import numpy as np

from .. import _native


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
    return _native.plane_intersect(
        np.ascontiguousarray(ro, dtype=float),
        np.ascontiguousarray(d, dtype=float),
        np.ascontiguousarray(o, dtype=float),
        np.ascontiguousarray(n, dtype=float),
        float(eps),
    )
