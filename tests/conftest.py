from pathlib import Path

import numpy as np
import pytest
import sympy as sp

import optics.refraction as refraction
from optics.config import (
    refractive_element_from_config,
    refractive_element_from_json,
)


def _parametric_equation(equation: str) -> str:
    x, y = sp.symbols("x y")
    u, v = sp.symbols("u v")
    expr = sp.parse_expr(equation)
    return str(expr.subs({x: u, y: v}))


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def sphere_schema(repo_root) -> str:
    return (repo_root / "tests/test_geometry.json").read_text()


@pytest.fixture
def sphere(sphere_schema) -> refraction.RefractiveElement:
    return refractive_element_from_json(sphere_schema)


@pytest.fixture
def offset_lens_schema(repo_root) -> str:
    return (repo_root / "tests/offset_geometry.json").read_text()


@pytest.fixture
def offset_lens(offset_lens_schema) -> refraction.RefractiveElement:
    return refractive_element_from_json(offset_lens_schema)


@pytest.fixture
def make_element():
    def _make(surface1: str, surface2: str, n: float = 1.5, orientation=None, aperture_radius=None) -> refraction.RefractiveElement:
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
