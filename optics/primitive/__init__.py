"""Geometric primitives: rays, transforms, hits, surfaces, apertures, sidewalls."""

from .aperture import Aperture, CircularAperture
from .hit import Hit, MeshHit, RayHit
from .material import Material
from .ray import Ray
from .sidewall import LensSidewall
from .surface import Surface
from .transform import Transform

__all__ = [
    "Aperture",
    "CircularAperture",
    "Hit",
    "LensSidewall",
    "Material",
    "MeshHit",
    "Ray",
    "RayHit",
    "Surface",
    "Transform",
]
