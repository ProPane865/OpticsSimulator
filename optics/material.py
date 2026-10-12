import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    refractive_index: float

    def __post_init__(self):
        n = float(self.refractive_index)
        if not math.isfinite(n) or n <= 0.0:
            raise ValueError("Refractive index must be finite and positive")
        object.__setattr__(self, "refractive_index", n)
