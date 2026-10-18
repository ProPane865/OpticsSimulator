"""Numerical tolerances and solver defaults shared across the optics core."""

import numpy as np

# --- Damped Newton iteration (solver) ---
MAX_NEWTON_ITER = 50
STEP_REL_TOL = 1e-14
CLAMP_REL = 0.5
CONVERGED_F_NORM = 1e-12
# Square Jacobians whose determinant is smaller than this fraction of the
# product of their column norms are treated as rank-deficient and solved in a
# least-squares sense (near-tangent / rim intersections).
NEWTON_SINGULAR_REL = 1e-10
# Smallest accepted fraction of a Newton step before the iteration is abandoned.
NEWTON_MIN_STEP = 1e-12
# A step damped below this fraction means the iteration is pinned against the
# edge of the domain; it is reported as stalled instead of crawling on.
NEWTON_STALL_STEP = 1e-3
# Backtracked (damped) iterations tolerated in total before the iteration is
# called stalled. Damping is what lets rim/tangent intersections converge, but
# runs that need many of them are almost never recoverable.
NEWTON_MAX_DAMPED_STEPS = 3

# Outcome codes of the damped Newton iteration, mirroring
# ``native/include/optics/common/constants.hpp``;
# they index the per-status tolerance tables below.
SOLVER_CONV = 0
SOLVER_STALL = 1
SOLVER_EXH = 2

# Per-status acceptance tolerances, indexed by the solver outcome codes.
PARAM_TOLS = np.array([1e-11, 1e-9, 1e-8])
HIT_TOLS = np.array([1e-11, 1e-10, 1e-10])
DEDUP_UV_TOL = 1e-6

# --- Seed grid for intersection / parameter inversion ---
SEED_GRID_N = 64
SEED_TOL = 0.75
SEED_TOL_FALLBACK = 2.0
MAX_SEEDS = 12
# Grid samples used to seed the local (x, y) -> (u, v) inversion.
XY_SEEDS = 12
POINT_MATCH_TOL = 1e-8

# --- Refraction / element tracing ---
RAY_EPSILON = 1e-9
MIN_HIT_T = 1e-7
MAX_INTERNAL_BOUNCES = 8
SIDEWALL_MATCH_ATOL = 1e-6
# Grid resolution used to bound an element's local volume for containment tests.
CONTAINMENT_BOUNDS_N = 16

# --- Aperture ---
APERTURE_EPS = 1e-12
APERTURE_BOUNDARY_N = 128

# --- Sidewall (Moller-Trumbore) intersection ---
DET_SINGULAR_EPS = 1e-14
BARY_TOL = 1e-12
MIN_FACE_T = 1e-9

# --- Vector helpers ---
# Cross-product magnitude below which two unit vectors count as parallel
# (or antiparallel) in the z-axis alignment rotation.
VECTOR_PARALLEL_TOL = 1e-12

# --- Surface domain / measurement plane ---
DOMAIN_TOL = 1e-9
PLANE_EPS = 1e-12
# Where the analytic derivative blows up (e.g. at an aperture rim, where the
# surface turns vertical), tangents are re-evaluated at a point pulled toward
# the interior of the parameter domain. These set the initial pull-back
# fraction of the domain span and the number of doublings attempted.
DERIV_INTERIOR_FRAC = 1e-6
DERIV_INTERIOR_TRIES = 8
# Tolerance on |x - u|, |y - v| when testing a surface for the (u, v, z(u, v))
# graph form, which allows reading a local height without solving for parameters.
GRAPH_XY_TOL = 1e-9
# --- Numerical derivatives (finite differences) -----------------------------
# Tangents are computed by central finite differences whose step is proportional
# to the parameter span: ``h = FD_STEP_REL * (u_max - u_min)`` (and likewise for
# ``v``), floored at ``FD_MIN_STEP`` so degenerate ranges never give a zero step.
# The same policy is mirrored in the native expression interpreter
# (``optics._native``); keep it in lock-step with the native constants header
# (``native/include/optics/common/constants.hpp``).
FD_STEP_REL = 1e-6
FD_MIN_STEP = 1e-12
# Shrink budget for the oriented one-sided differences used when a central
# sample leaves the parameter domain or is non-finite (e.g. across a sqrt edge).
FD_STEP_SHRINKS = 24
