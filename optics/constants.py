"""Numerical tolerances and solver defaults shared across the optics core."""

# --- Damped Newton iteration (solver) ---
MAX_NEWTON_ITER = 50
STEP_REL_TOL = 1e-14
CLAMP_REL = 0.5
CONVERGED_F_NORM = 1e-12

# Per-status acceptance tolerances, keyed by the solver status labels.
PARAM_TOL = {"converged": 1e-11, "stalled": 1e-9, "exhausted": 1e-8}
HIT_TOL = {"converged": 1e-11, "stalled": 1e-10, "exhausted": 1e-10}
DEDUP_UV_TOL = 1e-6

# --- Seed grid for intersection / parameter inversion ---
SEED_GRID_N = 64
SEED_TOL = 0.75
SEED_TOL_FALLBACK = 2.0
MAX_SEEDS = 12
POINT_MATCH_TOL = 1e-8

# --- Refraction / element tracing ---
RAY_EPSILON = 1e-9
MIN_HIT_T = 1e-7
MAX_INTERNAL_BOUNCES = 8
SIDEWALL_MATCH_ATOL = 1e-6

# --- Aperture ---
APERTURE_EPS = 1e-12
APERTURE_BOUNDARY_N = 128

# --- Sidewall (Moller-Trumbore) intersection ---
DET_SINGULAR_EPS = 1e-14
BARY_TOL = 1e-12
MIN_FACE_T = 1e-9

# --- Surface domain / measurement plane ---
DOMAIN_TOL = 1e-9
PLANE_EPS = 1e-12
