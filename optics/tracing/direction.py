"""Interface direction kernels: refraction and total internal reflection."""

import numpy as np
from numba import njit

from ..numerical.vector import normalize_vector


@njit(cache=True)
def _reflect_direction(incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return normalize_vector(incident - 2.0 * float(np.dot(normal, incident)) * normal)


@njit(cache=True)
def _refract_direction(
    incident: np.ndarray, normal: np.ndarray, eta: float, cos_i: float, k: float
) -> np.ndarray:
    return normalize_vector(eta * incident + (np.sqrt(k) - eta * cos_i) * normal)


@njit(cache=True)
def _orient(normal: np.ndarray, incident: np.ndarray) -> np.ndarray:
    """Align the surface normal with the incident ray direction."""
    if float(np.dot(normal, incident)) < 0.0:
        return -normal
    return normal


@njit(cache=True)
def refract_or_reflect(
    incident: np.ndarray, normal: np.ndarray, n_from: float, n_to: float
) -> "tuple[np.ndarray, bool]":
    """Return ``(direction, refracted)`` for one interface interaction.

    ``normal`` must be oriented so that ``dot(normal, incident) >= 0``.
    On total internal reflection the reflected direction is returned with
    ``refracted=False``.
    """
    eta = n_from / n_to
    cos_i = float(np.dot(normal, incident))
    k = 1.0 - eta**2 * (1.0 - cos_i**2)

    if k < 0.0:
        return _reflect_direction(incident, normal), False
    return _refract_direction(incident, normal, eta, cos_i, k), True


@njit(cache=True)
def trace_direction(incident: np.ndarray, normal: np.ndarray, n_from: float, n_to: float) -> np.ndarray:
    return refract_or_reflect(incident, normal, n_from, n_to)[0]
