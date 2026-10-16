import re
from pathlib import Path

import numpy as np
import pytest
from _reference import force_python_path as _force_python_path

import optics.tracing as tracing
from optics.config import (
    refractive_element_from_config,
    refractive_element_from_json,
)


def _parametric_equation(equation: str) -> str:
    # Test surface equations are written in spatial variables ``x``/``y``; the
    # surface parameterisation uses ``u``/``v``. Word-boundary replace so tokens
    # like ``exp`` are untouched.
    s = re.sub(r"\bx\b", "u", equation)
    return re.sub(r"\by\b", "v", s)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def force_python_path():
    """Return a callable that pins the given elements to the reference path.

    Replaces each element's compiled surfaces with pure-Python ``ReferenceSurface``
    clones (see ``tests/_reference.py``) and clears the element geometry caches.
    Every downstream entry point then falls back to the reference implementation:
    the reference solver (via ``njit_descriptor is None``), the serial element
    tracer, the Python stack traversal (``build_trace_context`` returns ``None``),
    and the serial measurement path. Used to differentially test the nopython
    kernels against this independent oracle.
    """

    def _force(elements) -> None:
        _force_python_path(elements)

    return _force


@pytest.fixture
def sphere_schema(repo_root) -> str:
    return (repo_root / "tests/test_geometry.json").read_text()


@pytest.fixture
def sphere(sphere_schema) -> tracing.RefractiveElement:
    return refractive_element_from_json(sphere_schema)


@pytest.fixture
def offset_lens_schema(repo_root) -> str:
    return (repo_root / "tests/offset_geometry.json").read_text()


@pytest.fixture
def offset_lens(offset_lens_schema) -> tracing.RefractiveElement:
    return refractive_element_from_json(offset_lens_schema)


@pytest.fixture
def make_element():
    def _make(
        surface1: str,
        surface2: str,
        n: float = 1.5,
        orientation=None,
        aperture_radius=None,
    ) -> tracing.RefractiveElement:
        orientation = (
            np.array([0.0, 0.0, 1.0])
            if orientation is None
            else np.asarray(orientation, dtype=float)
        )
        aperture = (
            None
            if aperture_radius is None
            else {"radius": float(aperture_radius)}
        )

        def _surface(z_eq: str) -> dict:
            entry = {
                "x": "u",
                "y": "v",
                "z": _parametric_equation(z_eq),
                "u_range": [-10.0, 10.0],
                "v_range": [-10.0, 10.0],
            }
            if aperture is not None:
                entry["aperture"] = aperture
            return entry

        config = {
            "surface1": _surface(surface1),
            "surface2": _surface(surface2),
            "material": {"refractive_index": n},
        }
        return refractive_element_from_config(config, orientation)

    return _make
