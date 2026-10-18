"""Native kernels inverting local ``(x, y)`` targets to surface parameters."""

from typing import Any

import numpy as np

from .. import _native


def _solve_xy_kernel(
    desc: Any, x_target: float, y_target: float, u: float, v: float
) -> "tuple[float, float, bool]":
    """Newton inversion of local ``(x, y)`` to parameters from one seed."""
    return _native.solve_xy(desc, float(x_target), float(y_target), float(u), float(v))


def parameters_at_xy_kernel(desc: Any, grid: Any, x: float, y: float) -> "tuple[float, float, bool]":
    """Grid-seeded local ``(x, y)`` inversion."""
    return _native.parameters_at_xy(desc, grid, float(x), float(y))


def parameters_at_kernel(desc: Any, grid: Any, point: Any) -> "tuple[float, float, bool]":
    """Local parameters whose surface point matches ``point`` (local space)."""
    return _native.parameters_at(desc, grid, point)


def build_rim_kernel(desc: Any, grid: Any, radius: float, cx: float, cy: float, n: int) -> np.ndarray:
    """Local-space circular-aperture rim with ``n`` points, shape ``(n, 3)``.

    Raises :class:`ValueError` when a rim point cannot be inverted onto the
    surface (the same failure the previous per-point loop raised).
    """
    return _native.build_rim(desc, grid, float(radius), float(cx), float(cy), int(n))
