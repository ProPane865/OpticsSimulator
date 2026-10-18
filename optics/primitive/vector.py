"""Vector helpers backed by the native kernels."""

from collections.abc import Sequence

import numpy as np

from .. import _native


def normalize_vector(v: np.ndarray) -> np.ndarray:
    return np.asarray(_native.normalize(np.ascontiguousarray(v, dtype=float)))


def normalize(vector: np.ndarray | Sequence[float]) -> np.ndarray:
    return normalize_vector(np.asarray(vector, dtype=float))


def rotation_from_z(direction: np.ndarray | Sequence[float]) -> np.ndarray:
    """Rodrigues rotation matrix mapping the +Z axis onto ``direction``.

    The rotation axis is the (unnormalized) cross product of +Z with the
    target direction; near-parallel targets need no rotation and
    antiparallel ones a 180-degree turn about X.
    """
    return np.asarray(
        _native.rotation_from_z(np.ascontiguousarray(direction, dtype=float))
    )
