"""Data layouts shared between the Python builders and the nopython kernels.

The nopython solver/tracer (``solver.py``, ``refraction.py``) passes surfaces,
seed grids and element stacks into ``@njit`` code as flat tuples of plain data.
Those tuples are indexed positionally inside nopython functions, where a wrong
index fails silently, so the layout is defined exactly once here: the integer
constants are the single source of truth for both the builders (Python side) and
the kernels (nopython side).

Numba resolves these module-level integers as compile-time tuple indices, so
using ``desc[D_ROT]`` instead of ``desc[11]`` costs nothing at runtime.
"""

from typing import Any

import numpy as np

# --- Surface descriptor (``Surface.njit_descriptor``) -----------------------
# A single surface flattened for the Newton/intersection kernels. The callables
# are ``numba.cfunc`` pointers with a fixed C signature, so every surface shares
# one argument type and the kernels compile (and disk-cache) once in total.
D_EVAL = 0    # cfunc(u, v, out_addr) -> local point
D_DERIV = 1   # cfunc(u, v, ru_addr, rv_addr) -> local tangents
D_U_MIN = 2
D_U_MAX = 3
D_V_MIN = 4
D_V_MAX = 5
D_TU = 6      # domain tolerance along u
D_TV = 7      # domain tolerance along v
D_AP_R = 8    # circular aperture radius (-1.0 = no aperture)
D_AP_CX = 9   # aperture center x
D_AP_CY = 10  # aperture center y
D_ROT = 11    # (3, 3) local->world rotation
D_TRANS = 12  # (3,) local->world translation


def build_descriptor(
    eval_cfn: Any,
    deriv_cfn: Any,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    tu: float,
    tv: float,
    ap_r: float,
    ap_cx: float,
    ap_cy: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> "tuple[Any, ...]":
    """Assemble a surface descriptor tuple in the layout above."""
    return (
        eval_cfn,
        deriv_cfn,
        u_min,
        u_max,
        v_min,
        v_max,
        tu,
        tv,
        ap_r,
        ap_cx,
        ap_cy,
        np.ascontiguousarray(rotation),
        np.ascontiguousarray(translation),
    )


# --- Seed grid (``SurfaceSolver.njit_grid``) --------------------------------
# A sampled parameter grid used to seed Newton for intersections and parameter
# inversion. ``*_valid`` arrays hold only in-domain/aperture points.
G_P_WORLD = 0   # (M, 3) valid sample points in world space
G_UF_VALID = 1  # (M,) u parameter per valid sample
G_VF_VALID = 2  # (M,) v parameter per valid sample
G_PTS = 3       # (N, 3) all sample points in local space
G_FINITE = 4    # (N,) bool: sample point is finite
G_UF = 5        # (N,) u parameter per sample
G_VF = 6        # (N,) v parameter per sample


def build_grid(
    p_world: np.ndarray,
    uf_valid: np.ndarray,
    vf_valid: np.ndarray,
    pts: np.ndarray,
    finite: np.ndarray,
    uf: np.ndarray,
    vf: np.ndarray,
) -> "tuple[Any, ...]":
    """Assemble a seed-grid tuple in the layout above (contiguous arrays)."""
    return tuple(
        np.ascontiguousarray(a)
        for a in (p_world, uf_valid, vf_valid, pts, finite, uf, vf)
    )


# --- Element context (``refraction.build_trace_context``) -------------------
# One closed refractive element flattened for the stack-tracing kernels.
E_DESC1 = 0     # front surface descriptor
E_DESC2 = 1     # rear surface descriptor
E_GRID1 = 2     # front seed grid
E_GRID2 = 3     # rear seed grid
E_INDEX = 4     # refractive index
E_LO = 5        # (3,) local AABB low corner
E_HI = 6        # (3,) local AABB high corner
E_TRI = 7       # (K, 3, 3) sidewall triangles
E_NORMALS = 8   # (K, 3) sidewall face normals
E_GRAPH1 = 9    # front surface is a graph over local XY
E_GRAPH2 = 10   # rear surface is a graph over local XY


def build_element_context(
    desc1: "tuple[Any, ...]",
    desc2: "tuple[Any, ...]",
    grid1: "tuple[Any, ...]",
    grid2: "tuple[Any, ...]",
    refractive_index: float,
    lo: np.ndarray,
    hi: np.ndarray,
    tri: np.ndarray,
    normals: np.ndarray,
    graph1: bool,
    graph2: bool,
) -> "tuple[Any, ...]":
    """Assemble an element-context tuple in the layout above."""
    return (
        desc1,
        desc2,
        grid1,
        grid2,
        refractive_index,
        np.ascontiguousarray(lo),
        np.ascontiguousarray(hi),
        np.ascontiguousarray(tri),
        np.ascontiguousarray(normals),
        bool(graph1),
        bool(graph2),
    )
