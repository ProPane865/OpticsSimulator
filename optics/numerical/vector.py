from collections.abc import Sequence

import numpy as np
from numba import njit

from .. import constants as _c


@njit(cache=True)
def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))

    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Vector must be finite and nonzero")

    return v / norm


def normalize(vector: np.ndarray | Sequence[float]) -> np.ndarray:
    return normalize_vector(np.asarray(vector, dtype=float))


def rotation_from_z(direction: np.ndarray | Sequence[float]) -> np.ndarray:
    """Rodrigues rotation matrix mapping the +Z axis onto ``direction``.

    The rotation axis is the (unnormalized) cross product of +Z with the
    target direction; near-parallel targets need no rotation and
    antiparallel ones a 180-degree turn about X.
    """
    z_axis = np.array([0.0, 0.0, 1.0])
    direction_norm = normalize(direction)
    axis = np.cross(z_axis, direction_norm)
    axis_norm = float(np.linalg.norm(axis))
    cos_theta = float(np.inner(z_axis, direction_norm))

    if axis_norm < _c.VECTOR_PARALLEL_TOL:
        if cos_theta < 0.0:
            return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        return np.eye(3)

    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - cos_theta) / axis_norm**2)
