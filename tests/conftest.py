import json
from pathlib import Path

import numpy as np
import pytest

import main


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def sphere_schema(repo_root) -> str:
    return (repo_root / "tests/test_geometry.json").read_text()


@pytest.fixture
def sphere(sphere_schema) -> main.RefractiveElement:
    return main.RefractiveElement(sphere_schema)


@pytest.fixture
def make_element():
    def _make(surface1: str, surface2: str, n: float = 1.5, orientation=None) -> main.RefractiveElement:
        orientation = (
            np.array([0.0, 0.0, 1.0])
            if orientation is None
            else np.asarray(orientation, dtype=float)
        )
        schema = json.dumps(
            {
                "surface1": {"equation": surface1},
                "surface2": {"equation": surface2},
                "material": {"refractive_index": n},
            }
        )
        return main.RefractiveElement(schema, orientation)

    return _make
