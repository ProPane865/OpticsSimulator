"""Small-matrix linear algebra helpers shared by the pure-Python paths.

``norm`` is the Python entry point for the native kernel; the Newton-step
solver below mirrors the native normal-equations fallback and is used by the
pure-Python reference oracle (``tests/_reference.py``).
"""

import math
from collections.abc import Sequence

import numpy as np

from .. import _native
from .. import constants as _c


def _newton_delta(J: np.ndarray, F: np.ndarray) -> np.ndarray | None:
    """Solve ``J delta = -F``, falling back to least squares when rank-deficient."""
    try:
        if J.shape[0] == J.shape[1] and _is_nonsingular(J):
            return np.linalg.solve(J, -F)
        return np.linalg.lstsq(J, -F, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None


def _is_nonsingular(J: np.ndarray) -> bool:
    """Whether ``det(J)`` is large relative to the product of ``J``'s column norms."""
    rows, cols = J.shape
    if rows != cols:
        return False
    if rows == 3:
        a = J[0, 0]
        b = J[0, 1]
        c = J[0, 2]
        d = J[1, 0]
        e = J[1, 1]
        f = J[1, 2]
        g = J[2, 0]
        h = J[2, 1]
        i = J[2, 2]
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        scale = (
            math.sqrt(a * a + d * d + g * g)
            * math.sqrt(b * b + e * e + h * h)
            * math.sqrt(c * c + f * f + i * i)
        )
    elif rows == 2:
        a = J[0, 0]
        b = J[0, 1]
        c = J[1, 0]
        d = J[1, 1]
        det = a * d - b * c
        scale = math.sqrt(a * a + c * c) * math.sqrt(b * b + d * d)
    else:
        return False
    if scale <= 0.0 or not math.isfinite(scale):
        return False
    return bool(abs(det) / scale > _c.NEWTON_SINGULAR_REL)


def norm(v: np.ndarray | Sequence[float]) -> float:
    """Euclidean norm (native kernel entry point)."""
    return float(_native.norm(np.ascontiguousarray(v, dtype=float)))
