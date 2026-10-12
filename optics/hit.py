from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RayHit:
    point: np.ndarray
    t: float


@dataclass(frozen=True)
class Hit(RayHit):
    u: float | None = None
    v: float | None = None


@dataclass(frozen=True)
class MeshHit(RayHit):
    face: int
    barycentric: np.ndarray
