import numpy as np

from dataclasses import dataclass

from .vector import normalize

@dataclass(frozen=True)
class Ray:
    origin: np.ndarray
    direction: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "direction", normalize(self.direction))

    def at(self, t):
        return self.origin + t * self.direction