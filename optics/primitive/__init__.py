"""Geometric primitives: rays, transforms, hits, surfaces, apertures, sidewalls."""

from .aperture import CircularAperture
from .hit import Hit, MeshHit
from .material import Material
from .ray import Ray
from .sidewall import LensSidewall
from .surface import Surface
from .transform import Transform

__all__ = [
    "CircularAperture",
    "Hit",
    "LensSidewall",
    "Material",
    "MeshHit",
    "Ray",
    "Surface",
    "Transform",
]
