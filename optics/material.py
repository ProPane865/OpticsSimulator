from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    refractive_index: float
