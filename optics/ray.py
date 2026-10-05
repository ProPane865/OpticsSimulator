import numpy as np

from .vector import normalize

class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = np.array(origin, dtype=float)
        self.direction = normalize(direction)