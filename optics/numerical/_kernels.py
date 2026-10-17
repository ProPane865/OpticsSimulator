"""Data layouts shared between the Python builders and the native kernels.

The native solver/tracer (``optics._native``) passes surfaces, seed grids and
element stacks as opaque native objects built here from plain numpy arrays;
the layouts (a surface descriptor, a seed grid and an element context) are the
C++ counterparts of the old flat-tuple protocol from the Numba kernels, with
shape validation confined to these builders.
"""

from typing import Any

import numpy as np

from .. import _native


def build_descriptor(
    model: Any,
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
) -> Any:
    """Assemble a native surface descriptor from a compiled model."""
    return _native.build_surface_desc(
        model,
        float(u_min),
        float(u_max),
        float(v_min),
        float(v_max),
        float(tu),
        float(tv),
        float(ap_r),
        float(ap_cx),
        float(ap_cy),
        np.ascontiguousarray(rotation, dtype=float),
        np.ascontiguousarray(translation, dtype=float),
    )


def build_grid(
    p_world: np.ndarray,
    uf_valid: np.ndarray,
    vf_valid: np.ndarray,
    pts: np.ndarray,
    finite: np.ndarray,
    uf: np.ndarray,
    vf: np.ndarray,
) -> Any:
    """Assemble a native seed grid (contiguous arrays)."""
    assert p_world.shape[0] == uf_valid.size == vf_valid.size
    assert pts.shape[0] == finite.size == uf.size == vf.size
    return _native.build_grid(
        np.ascontiguousarray(p_world, dtype=float),
        np.ascontiguousarray(uf_valid, dtype=float),
        np.ascontiguousarray(vf_valid, dtype=float),
        np.ascontiguousarray(pts, dtype=float),
        np.ascontiguousarray(finite, dtype=bool),
        np.ascontiguousarray(uf, dtype=float),
        np.ascontiguousarray(vf, dtype=float),
    )


def build_element_context(
    desc1: Any,
    desc2: Any,
    grid1: Any,
    grid2: Any,
    refractive_index: float,
    lo: np.ndarray,
    hi: np.ndarray,
    tri: np.ndarray,
    normals: np.ndarray,
    graph1: bool,
    graph2: bool,
) -> Any:
    """Assemble a native element context (front/rear surfaces + sidewall)."""
    return _native.build_element_context(
        desc1,
        desc2,
        grid1,
        grid2,
        float(refractive_index),
        np.ascontiguousarray(lo, dtype=float),
        np.ascontiguousarray(hi, dtype=float),
        np.ascontiguousarray(tri, dtype=float),
        np.ascontiguousarray(normals, dtype=float),
        bool(graph1),
        bool(graph2),
    )


def build_trace_stack(contexts: list[Any]) -> Any:
    """Assemble a native trace stack from element contexts."""
    return _native.build_trace_stack(list(contexts))
