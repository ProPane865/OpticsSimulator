"""Data layouts shared between the Python builders and the native kernels.

The native solver/tracer (``optics._native``) passes surfaces, seed grids and
element stacks as opaque native objects built here from plain numpy arrays;
shape validation is confined to these builders.
"""

from typing import Any

import numpy as np

from .. import _native


def build_seed_grid(desc: Any, grid_n: int) -> Any:
    """Assemble a native seed grid directly from a native surface descriptor."""
    return _native.build_seed_grid(desc, int(grid_n))


def make_surface(
    model: Any,
    x_src: str,
    y_src: str,
    z_src: str,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    ap_r: float,
    ap_cx: float,
    ap_cy: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    grid_n: int,
) -> Any:
    """Build a persistent native surface (descriptor + seed grid + graph flag).

    The seed grid is sampled once at construction, so every later kernel call
    reuses it; there is no per-trace descriptor/grid rebuilding.
    """
    return _native.make_surface(
        model,
        str(x_src),
        str(y_src),
        str(z_src),
        float(u_min),
        float(u_max),
        float(v_min),
        float(v_max),
        float(ap_r),
        float(ap_cx),
        float(ap_cy),
        np.ascontiguousarray(rotation, dtype=float),
        np.ascontiguousarray(translation, dtype=float),
        int(grid_n),
    )


def build_element(
    surface1: Any,
    surface2: Any,
    refractive_index: float,
    tri: np.ndarray,
    normals: np.ndarray,
) -> Any:
    """Assemble a persistent native element (two surfaces + sidewall).

    The element's axis-aligned bounds and graph-over-xy flags are derived
    natively from the two native surfaces at construction, so they are not
    passed in; every later trace reuses this one object.
    """
    return _native.build_element(
        surface1,
        surface2,
        float(refractive_index),
        np.ascontiguousarray(tri, dtype=float),
        np.ascontiguousarray(normals, dtype=float),
    )


def build_trace_stack(elements: list[Any]) -> Any:
    """Assemble a native trace stack from persistent native elements."""
    return _native.build_trace_stack(list(elements))
