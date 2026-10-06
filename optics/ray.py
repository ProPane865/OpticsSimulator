import numpy as np

from dataclasses import dataclass

from .vector import normalize

class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin
        self.direction = normalize(direction)

        if self.origin.shape != (3,):
            raise ValueError("Origin must be array of shape (3,)")

        if self.direction.shape != (3,):
            raise ValueError("Direction must be array of shape (3,)")

    def at(self, t):
        return self.origin + t * self.direction