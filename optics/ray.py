import numpy as np

from dataclasses import dataclass

from .vector import normalize

class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray, power: float):
        self.origin = origin
        self.direction = normalize(direction)
        self.power = power

        if self.origin.shape != (3,):
            raise ValueError("Origin must be array of shape (3,)")

        if self.direction.shape != (3,):
            raise ValueError("Direction must be array of shape (3,)")

        if self.power < 0:
            raise ValueError("Power must be nonnegative")

    def at(self, t):
        return self.origin + t * self.direction