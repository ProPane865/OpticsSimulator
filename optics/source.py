from collections.abc import Sequence

import numpy as np

from .primitive.ray import Ray

SINGLE_POINT = "single_point"
POINT_ARRAY = "point_array"

_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def fibonacci_cone_directions(count: int, cone_angle_deg: float = 360.0) -> np.ndarray:
    count = int(count)
    if count <= 0:
        return np.empty((0, 3), dtype=float)
    cone_angle_deg = float(cone_angle_deg)
    if not np.isfinite(cone_angle_deg) or not 0.0 <= cone_angle_deg <= 360.0:
        raise ValueError("Cone angle must be finite and between 0 and 360 degrees")

    half_angle = np.deg2rad(cone_angle_deg / 2.0)
    cos_max = np.cos(half_angle)
    i = np.arange(count, dtype=float)
    c = cos_max + (1.0 - cos_max) * (i + 0.5) / count
    r = np.sqrt(np.maximum(0.0, 1.0 - c * c))
    phi = i * _GOLDEN_ANGLE
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), -c])


def fibonacci_sphere_directions(count: int) -> np.ndarray:
    return fibonacci_cone_directions(count, 360.0)


class PointSource:
    position: np.ndarray
    power: float
    num_rays: int
    source_type: str
    cone_angle: float

    def __init__(
        self,
        position: np.ndarray | Sequence[float],
        power: float = 1.0,
        num_rays: int = 64,
        source_type: str = SINGLE_POINT,
        cone_angle: float = 360.0,
    ) -> None:
        self.position = np.asarray(position, dtype=float)
        if self.position.shape != (3,):
            raise ValueError("Position must be array of shape (3,)")
        if not np.all(np.isfinite(self.position)):
            raise ValueError("Position must be finite")

        self.power = float(power)
        if not np.isfinite(self.power) or self.power < 0.0:
            raise ValueError("Power must be finite and non-negative")

        self.num_rays = int(num_rays)
        if self.num_rays < 1:
            raise ValueError("Rays per point must be at least 1")

        self.source_type = str(source_type)

        self.cone_angle = float(cone_angle)
        if not np.isfinite(self.cone_angle) or not 0.0 <= self.cone_angle <= 360.0:
            raise ValueError("Cone angle must be finite and between 0 and 360 degrees")

    def emit_rays(self) -> list[Ray]:
        if self.source_type == POINT_ARRAY:
            raise NotImplementedError("Point array sources are not implemented")
        if self.source_type != SINGLE_POINT:
            raise ValueError(f"Unknown source type: {self.source_type}")

        directions = fibonacci_cone_directions(self.num_rays, self.cone_angle)
        ray_power = self.power / self.num_rays
        return [Ray(self.position.copy(), d, ray_power) for d in directions]
