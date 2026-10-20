"""Pure-Python reference implementations used only by the test/golden suites.

The native kernels (``optics._native``) implement the production ray sources and
numerics; the helpers here re-derive the same quantities in NumPy so tests can
check the native results against an independent oracle. They are deliberately
kept out of the shipped :mod:`optics` public API.
"""

import numpy as np

_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def fibonacci_cone_directions(count: int, cone_angle_deg: float = 360.0) -> np.ndarray:
    """Unit directions spread over a cone by Fibonacci-sphere spacing.

    A pure-Python reference fan generator for tests and golden capture; the
    native ``emit_cone`` kernel produces the production ray fan.
    """
    count = int(count)
    if count <= 0:
        return np.empty((0, 3), dtype=float)
    cone_angle_deg = float(cone_angle_deg)
    if not np.isfinite(cone_angle_deg) or not 0.0 <= cone_angle_deg <= 360.0:
        raise ValueError("Cone angle must be finite and between 0 and 360 degrees")

    half_angle = np.deg2rad(cone_angle_deg / 2.0)
    cos_max = np.cos(half_angle)
    i = np.arange(count, dtype=float)
    c = cos_max + (1.0 - cos_max) * (i + 0.5) / count
    r = np.sqrt(np.maximum(0.0, 1.0 - c * c))
    phi = i * _GOLDEN_ANGLE
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), -c])
