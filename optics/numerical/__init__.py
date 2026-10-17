"""Numerical kernels and solvers: linear algebra, Newton iteration,
ray/surface intersection, parameter inversion, vector helpers.

The hot paths run in the compiled native module (``optics._native``); this
package is a dependency-free leaf layer of :mod:`optics`: it only depends on
:mod:`optics.constants`, never on :mod:`optics.primitive`.
"""

from ._kernels import build_descriptor, build_element_context, build_grid
from .desc import build_surface_descriptor, probe_njit_pair
from .intersect import surface_intersect_njit
from .invert import parameters_at_xy_njit
from .linalg import norm
from .mesh import mesh_intersect_kernel
from .parallel import run_in_chunks
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
    "norm",
    "parameters_at_xy_njit",
    "plane_intersect_kernel",
    "probe_njit_pair",
    "rotation_from_z",
    "run_in_chunks",
    "surface_intersect_njit",
]
