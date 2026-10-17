"""Native kernels inverting local ``(x, y)`` targets to surface parameters."""

from typing import Any

from .. import _native


def _solve_xy_njit(
    desc: Any, x_target: float, y_target: float, u: float, v: float
) -> "tuple[float, float, bool]":
    """Newton inversion of local ``(x, y)`` to parameters from one seed."""
    return _native.solve_xy(desc, float(x_target), float(y_target), float(u), float(v))


def parameters_at_xy_njit(desc: Any, grid: Any, x: float, y: float) -> "tuple[float, float, bool]":
    """Grid-seeded local ``(x, y)`` inversion."""
    return _native.parameters_at_xy(desc, grid, float(x), float(y))
