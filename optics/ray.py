import numpy as np

from .vector import normalize


class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray, power: float):
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
        power = float(power)

        if origin.shape != (3,):
            raise ValueError("Origin must be array of shape (3,)")

        if not np.all(np.isfinite(origin)):
            raise ValueError("Ray origin must be finite")

        if direction.shape != (3,):
            raise ValueError("Direction must be array of shape (3,)")

        if not np.all(np.isfinite(direction)):
            raise ValueError("Ray direction must be finite")

        if np.linalg.norm(direction) < 1e-12:
            raise ValueError("Ray direction must be nonzero")

        if not np.isfinite(power):
            raise ValueError("Ray power must be finite")

        if power < 0:
            raise ValueError("Power cannot be negative")

        self.origin = origin
        self.direction = normalize(direction)
        self.power = power

    def at(self, t):
        return self.origin + t * self.direction