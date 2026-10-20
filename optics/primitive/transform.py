from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transform:
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float)

        if rotation.shape != (3, 3):
            raise ValueError("Rotation must be a 3x3 matrix")
        if translation.shape != (3,):
            raise ValueError("Translation must be an array of shape (3,)")
        if not (np.all(np.isfinite(rotation)) and np.all(np.isfinite(translation))):
            raise ValueError("Transform entries must be finite")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8):
            raise ValueError("Rotation must be orthonormal")

        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def point_to_world(self, p: np.ndarray) -> np.ndarray:
        return self.rotation @ p + self.translation

    def point_to_local(self, p: np.ndarray) -> np.ndarray:
        return self.rotation.T @ (p - self.translation)

    def points_to_world(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).T
        return (self.rotation @ pts + self.translation[:, None]).T

    def vector_to_world(self, v: np.ndarray) -> np.ndarray:
        return self.rotation @ v

    def vector_to_local(self, v: np.ndarray) -> np.ndarray:
        return self.rotation.T @ v
