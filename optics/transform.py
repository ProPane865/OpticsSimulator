from dataclasses import dataclass
from .ray import Ray

import numpy as np

@dataclass(frozen=True)
class Transform:
    rotation: np.ndarray
    translation: np.ndarray

    def point_to_world(self, p):
        return self.rotation @ p + self.translation

    def point_to_local(self, p):
        return self.rotation.T @ (p - self.translation)

    def points_to_world(self, points):
        pts = np.asarray(points, dtype=float).T
        return (self.rotation @ pts + self.translation[:, None]).T

    def vector_to_world(self, v):
        return self.rotation @ v

    def vector_to_local(self, v):
        return self.rotation.T @ v

    def ray_to_local(self, ray):
        return Ray(
            self.point_to_local(ray.origin),
            self.vector_to_local(ray.direction),
        )