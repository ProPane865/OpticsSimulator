"""Numerical kernels and solvers: ray/surface intersection, parameter
inversion, plane and mesh intersection, vector helpers.

The hot paths run in the compiled native module (``optics._native``); this
package is a dependency-free leaf layer of :mod:`optics`: it only depends on
:mod:`optics.constants`, never on :mod:`optics.primitive`.
"""

from ._kernels import build_descriptor, build_element_context, build_grid
from .desc import build_surface_descriptor
from .intersect import surface_intersect_kernel
from .invert import parameters_at_xy_kernel
from .mesh import mesh_intersect_kernel
from .plane import plane_intersect_kernel
from .solver import SurfaceSolver
from .vector import normalize, normalize_vector, rotation_from_z

__all__ = [
    "SurfaceSolver",
    "build_descriptor",
    "build_element_context",
    "build_grid",
    "build_surface_descriptor",
    "mesh_intersect_kernel",
    "normalize",
    "normalize_vector",
    "parameters_at_xy_kernel",
    "plane_intersect_kernel",
    "rotation_from_z",
    "surface_intersect_kernel",
]
