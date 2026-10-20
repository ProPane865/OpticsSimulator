"""Numerical tolerances shared between the Python layer and the native kernels.

The solver-side tolerances (Newton iteration, seed grids, finite differences)
live in the native constants header
(``native/include/optics/common/constants.hpp``); only the values the Python
layer itself consumes are mirrored here.
"""

# --- Intersection setup ------------------------------------------------------
# Grid resolution of the seed grid sampled once per native surface.
SEED_GRID_N = 64

# --- Tracing -----------------------------------------------------------------
# Ceiling on internal (total-internal-reflection) bounces inside a stack;
# bounds the per-element segment capacity of the trace buffers.
MAX_INTERNAL_BOUNCES = 8
# Tolerance for deciding that two aperture rims are too close to warrant a
# sidewall between them.
SIDEWALL_MATCH_ATOL = 1e-6

# --- Aperture ----------------------------------------------------------------
APERTURE_EPS = 1e-12
APERTURE_BOUNDARY_N = 128

# --- Sidewall (Moller-Trumbore) intersection ---------------------------------
DET_SINGULAR_EPS = 1e-14
BARY_TOL = 1e-12
MIN_FACE_T = 1e-9

# --- Surface domain / measurement plane ---------------------------------------
DOMAIN_TOL = 1e-9
PLANE_EPS = 1e-12
