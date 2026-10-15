"""Small-matrix linear algebra helpers shared by the Newton paths."""

import math

import numpy as np
from numba import njit

from .. import constants as _c


def _newton_delta(J: np.ndarray, F: np.ndarray) -> np.ndarray | None:
    """Solve ``J delta = -F``, falling back to least squares when rank-deficient."""
    try:
        if J.shape[0] == J.shape[1] and _is_nonsingular(J):
            return np.linalg.solve(J, -F)
        return np.linalg.lstsq(J, -F, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None


@njit(cache=True)
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


@njit(cache=True)
def norm(v: np.ndarray) -> float:
    """Euclidean norm.

    ``np.linalg.norm`` is avoided in nopython paths because its typing pass
    does not see through ``is None`` narrowing of optional arrays.
    """
    return float(math.sqrt(float(np.dot(v, v))))


@njit(cache=True)
def _solve_small(A: np.ndarray, b: np.ndarray) -> "np.ndarray | None":
    """Solve a 2x2/3x3 system by Gaussian elimination with partial pivoting."""
    n = A.shape[0]
    M = A.copy()
    y = b.copy()
    for k in range(n):
        p = k
        maxv = abs(M[k, k])
        for i in range(k + 1, n):
            if abs(M[i, k]) > maxv:
                maxv = abs(M[i, k])
                p = i
        if not math.isfinite(maxv) or maxv <= 0.0:
            return None
        if p != k:
            for j in range(n):
                tmp = M[k, j]
                M[k, j] = M[p, j]
                M[p, j] = tmp
            tmp = y[k]
            y[k] = y[p]
            y[p] = tmp
        for i in range(k + 1, n):
            f = M[i, k] / M[k, k]
            if f != 0.0:
                for j in range(k, n):
                    M[i, j] -= f * M[k, j]
                y[i] -= f * y[k]
    x = np.empty(n)
    for k in range(n - 1, -1, -1):
        s = y[k]
        for j in range(k + 1, n):
            s -= M[k, j] * x[j]
        x[k] = s / M[k, k]
    if not np.all(np.isfinite(x)):
        return None
    return x


@njit(cache=True)
def _newton_delta_njit(J: np.ndarray, F: np.ndarray) -> "np.ndarray | None":
    """Numpy-free ``_newton_delta``: square solve, else normal equations."""
    rows, cols = J.shape
    if rows == cols and (rows == 2 or rows == 3):
        if _is_nonsingular(J):
            return _solve_small(J, -F)
    if cols < 2 or cols > 3 or rows > 3:
        return None
    JtJ = J.T @ J
    if not _is_nonsingular(JtJ):
        return None
    return _solve_small(JtJ, J.T @ (-F))
