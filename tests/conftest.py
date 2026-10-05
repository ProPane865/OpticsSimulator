import json
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

import optics.refraction as refraction


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
    return refraction.RefractiveElement(sphere_schema)


@pytest.fixture
def make_element():
    def _make(surface1: str, surface2: str, n: float = 1.5, orientation=None) -> refraction.RefractiveElement:
        orientation = (
            np.array([0.0, 0.0, 1.0])
            if orientation is None
            else np.asarray(orientation, dtype=float)
        )
        schema = json.dumps(
            {
                "surface1": {
                    "x": "u",
                    "y": "v",
                    "z": _parametric_equation(surface1),
                    "u_range": [-10.0, 10.0],
                    "v_range": [-10.0, 10.0],
                },
                "surface2": {
                    "x": "u",
                    "y": "v",
                    "z": _parametric_equation(surface2),
                    "u_range": [-10.0, 10.0],
                    "v_range": [-10.0, 10.0],
                },
                "material": {"refractive_index": n},
            }
        )
        return refraction.RefractiveElement(schema, orientation)

    return _make
