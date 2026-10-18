from dataclasses import dataclass

import numpy as np

from .vector import normalize


@dataclass(frozen=True, eq=False)
class Ray:
    origin: np.ndarray
    direction: np.ndarray
    power: float

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=float)
        direction = np.asarray(self.direction, dtype=float)
        power = float(self.power)

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

        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "direction", normalize(direction))
        object.__setattr__(self, "power", power)

    def at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction
