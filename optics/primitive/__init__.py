"""Geometric primitives: rays, transforms, hits, surfaces, apertures, sidewalls."""

from .aperture import CircularAperture
from .material import Material
from .ray import Ray
from .sidewall import LensSidewall
from .surface import Surface
from .transform import Transform

__all__ = [
    "CircularAperture",
    "LensSidewall",
    "Material",
    "Ray",
    "Surface",
    "Transform",
]
