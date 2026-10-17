"""Native ray/surface intersection kernels (Python entry points)."""

from typing import Any

import numpy as np

from .. import _native


def _newton_intersection_njit(
    desc: Any,
    o: np.ndarray,
    d: np.ndarray,
    u: float,
    v: float,
    t: float,
) -> "tuple[float, float, float, bool]":
    """Full ray/surface Newton solve; returns ``(t, u, v, ok)``."""
    return _native.newton_intersection(
        desc,
        np.ascontiguousarray(o, dtype=float),
        np.ascontiguousarray(d, dtype=float),
        float(u),
        float(v),
        float(t),
    )


def surface_intersect_njit(
    desc: Any,
    grid: Any,
    o: np.ndarray,
    d: np.ndarray,
) -> "tuple[float, float, float, bool]":
    """Seed selection + Newton + dedup, the whole ray/surface intersection.

    Counterpart of ``SurfaceSolver.intersect``; returns the nearest forward
    hit as ``(t, u, v, ok)``.
    """
    return _native.surface_intersect(
        desc,
        grid,
        np.ascontiguousarray(o, dtype=float),
        np.ascontiguousarray(d, dtype=float),
    )
