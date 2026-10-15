"""Differential tests: nopython kernels vs the pure-Python reference path.

The nopython solver/tracer is the production path, but its only oracle is the
hand-maintained Python twin in ``solver.py`` / ``refraction.py``. These tests
run the same rays through both paths (the Python one forced via the
``force_python_path`` fixture) and assert they agree, so a divergence in either
implementation -- or a wrong descriptor/grid/element tuple index -- fails here
rather than shipping silently.
"""

import json

import numpy as np
import pytest

from optics import refraction as rf
from optics.config import refractive_element_from_json, stack_from_config
from optics.measurement_plane import MeasurementPlane
from optics.primitive.ray import Ray
from optics.primitive.surface import Surface

BICONE_S1 = "3 - 0.3*sqrt(x**2 + y**2)"
BICONE_S2 = "-1 + 2*sqrt(x**2 + y**2)"

POINT_ATOL = 1e-6
DIR_ATOL = 1e-7


# --- Ray sets ---------------------------------------------------------------
def _fan(n=16, seed=1, z=4.0, spread=0.5, tilt=0.05):
    rng = np.random.default_rng(seed)
    rays = []
    for _ in range(n):
        x, y = rng.uniform(-spread, spread, 2)
        d = np.array([rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), -1.0])
        rays.append(Ray(np.array([x, y, z]), d, 1.0))
    return rays


def _tir_rays():
    return [
        Ray(np.array([x, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), 1.0)
        for x in np.linspace(0.02, 0.2, 5)
    ]


def _sidewall_rays():
    rays = []
    for y in np.linspace(-0.8, 0.8, 5):
        for z in np.linspace(-1.5, -0.5, 3):
            rays.append(Ray(np.array([3.0, y, z]), np.array([-1.0, 0.0, 0.0]), 1.0))
    return rays


# --- Scenario builders ------------------------------------------------------
def _load_stack(repo_root, name):
    doc = json.loads((repo_root / "optics" / "examples" / name).read_text())
    return stack_from_config(doc["elements"])


def _scenario(name, make_element, repo_root):
    """Return ``(make_elements, rays)`` for a named parity scenario.

    ``make_elements`` is a zero-arg factory so each path (nopython and Python)
    can be exercised on freshly built, independent element objects.
    """
    if name == "sphere":
        schema = (repo_root / "tests" / "test_geometry.json").read_text()
        return lambda: [refractive_element_from_json(schema)], _fan()
    if name == "offset":
        schema = (repo_root / "tests" / "offset_geometry.json").read_text()
        return lambda: [refractive_element_from_json(schema)], _fan(spread=0.3)
    if name == "slab":
        return lambda: [make_element("0", "-2", n=1.5)], _fan(z=3.0)
    if name == "bicone":
        return lambda: [make_element(BICONE_S1, BICONE_S2, n=1.5)], _tir_rays()
    if name == "sidewall":
        return (
            lambda: [make_element("0", "-2", n=1.5, aperture_radius=1.0)],
            _sidewall_rays(),
        )
    if name == "tilted":
        def make():
            return [
                make_element(
                    "sqrt(1 - (x**2) - (y**2))",
                    "-sqrt(1 - (x**2) - (y**2))",
                    n=1.5,
                    orientation=[0.0, 1.0, 0.0],
                )
            ]

        rays = [
            Ray(np.array([0.0, 2.5, z]), np.array([0.0, -1.0, 0.0]), 1.0)
            for z in np.linspace(-0.4, 0.4, 5)
        ]
        return make, rays
    if name == "doublet":
        return lambda: _load_stack(repo_root, "doublet.json"), _fan(z=6.0, spread=0.4)
    if name == "triplet":
        return lambda: _load_stack(repo_root, "triplet.json"), _fan(z=6.0, spread=0.4)
    raise AssertionError(f"unknown scenario {name}")


SCENARIOS = [
    "sphere",
    "offset",
    "slab",
    "bicone",
    "sidewall",
    "tilted",
    "doublet",
    "triplet",
]


def _both(make_elements, rays, fn, force):
    """Run ``fn(elements, rays)`` on the nopython and Python paths separately."""
    fast = fn(make_elements(), rays)
    slow_elements = make_elements()
    force(slow_elements)
    slow = fn(slow_elements, rays)
    return fast, slow


def _assert_hit_equal(a, b):
    if a is None or b is None:
        assert a is None and b is None, f"hit/miss mismatch: {a!r} vs {b!r}"
        return
    np.testing.assert_allclose(a.point, b.point, atol=POINT_ATOL)
    assert a.t == pytest.approx(b.t, abs=POINT_ATOL)


# --- API-level parity -------------------------------------------------------
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_intersect_parity(scenario, make_element, repo_root, force_python_path):
    make, rays = _scenario(scenario, make_element, repo_root)

    def run(els, rs):
        return [
            [surface.intersect(ray) for ray in rs]
            for surface in (els[0].surface1, els[0].surface2)
        ]

    fast, slow = _both(make, rays, run, force_python_path)
    for fast_surf, slow_surf in zip(fast, slow):
        for fast_hit, slow_hit in zip(fast_surf, slow_surf):
            _assert_hit_equal(fast_hit, slow_hit)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_first_hit_parity(scenario, make_element, repo_root, force_python_path):
    make, rays = _scenario(scenario, make_element, repo_root)

    def run(els, rs):
        return [els[0].first_hit(ray) for ray in rs]

    fast, slow = _both(make, rays, run, force_python_path)
    for fast_hit, slow_hit in zip(fast, slow):
        if fast_hit is None or slow_hit is None:
            assert fast_hit is None and slow_hit is None
            continue
        assert fast_hit[0] == pytest.approx(slow_hit[0], abs=POINT_ATOL)
        np.testing.assert_allclose(fast_hit[1].point, slow_hit[1].point, atol=POINT_ATOL)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_trace_parity(scenario, make_element, repo_root, force_python_path):
    make, rays = _scenario(scenario, make_element, repo_root)

    def run(els, rs):
        return [els[0].trace(ray) for ray in rs]

    fast, slow = _both(make, rays, run, force_python_path)
    for fast_res, slow_res in zip(fast, slow):
        assert len(fast_res.internal_path) == len(slow_res.internal_path)
        assert (fast_res.exit_hit is None) == (slow_res.exit_hit is None)
        assert (fast_res.outgoing_ray is None) == (slow_res.outgoing_ray is None)
        if fast_res.outgoing_ray is not None:
            np.testing.assert_allclose(
                fast_res.outgoing_ray.origin, slow_res.outgoing_ray.origin, atol=POINT_ATOL
            )
            np.testing.assert_allclose(
                fast_res.outgoing_ray.direction, slow_res.outgoing_ray.direction, atol=DIR_ATOL
            )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_stack_path_parity(scenario, make_element, repo_root, force_python_path):
    make, rays = _scenario(scenario, make_element, repo_root)

    def run(els, rs):
        return [rf.trace_stack_path(ray, els, ray_length=2.0) for ray in rs]

    fast, slow = _both(make, rays, run, force_python_path)
    for (fast_segs, fast_final), (slow_segs, slow_final) in zip(fast, slow):
        assert len(fast_segs) == len(slow_segs)
        for (a0, b0, k0), (a1, b1, k1) in zip(fast_segs, slow_segs):
            assert k0 == k1
            np.testing.assert_allclose(a0, a1, atol=POINT_ATOL)
            np.testing.assert_allclose(b0, b1, atol=POINT_ATOL)
        assert (fast_final is None) == (slow_final is None)
        if fast_final is not None:
            np.testing.assert_allclose(fast_final.origin, slow_final.origin, atol=POINT_ATOL)
            np.testing.assert_allclose(
                fast_final.direction, slow_final.direction, atol=DIR_ATOL
            )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_trace_stack_final_parity(scenario, make_element, repo_root, force_python_path):
    make, rays = _scenario(scenario, make_element, repo_root)

    def run(els, rs):
        return [rf.trace_stack(ray, els) for ray in rs]

    fast, slow = _both(make, rays, run, force_python_path)
    for fast_out, slow_out in zip(fast, slow):
        assert (fast_out is None) == (slow_out is None)
        if fast_out is not None:
            np.testing.assert_allclose(fast_out.origin, slow_out.origin, atol=POINT_ATOL)
            np.testing.assert_allclose(fast_out.direction, slow_out.direction, atol=DIR_ATOL)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_measure_parity(scenario, make_element, repo_root, force_python_path):
    make, rays = _scenario(scenario, make_element, repo_root)

    def run(els, rs):
        plane = MeasurementPlane(
            origin=np.array([0.0, 0.0, -6.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            u_range=(-2.0, 2.0),
            v_range=(-2.0, 2.0),
            resolution=(16, 16),
        )
        return plane.measure(rs, els), plane.grid

    (fast_total, fast_grid), (slow_total, slow_grid) = _both(
        make, rays, run, force_python_path
    )
    assert fast_total == pytest.approx(slow_total, rel=1e-9, abs=1e-9)
    np.testing.assert_allclose(fast_grid, slow_grid, rtol=1e-9, atol=1e-9)


# --- Targeted parity for the remaining dual implementations -----------------
class _SurfaceHolder:
    """Adapter so ``force_python_path`` (which reads ``.surface1``/``.surface2``)
    can pin a bare :class:`Surface` to the Python path."""

    def __init__(self, surface):
        self.surface1 = surface
        self.surface2 = surface


def test_contains_point_parity(make_element, force_python_path):
    elem = make_element("2", "0", n=1.5, aperture_radius=1.0)
    points = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, -1.0],
        [2.0, 0.0, 1.0],
        [0.5, 0.5, 1.0],
        [0.9, 0.0, 1.0],
    ]
    fast = [elem.contains_point(p) for p in points]
    force_python_path([elem])
    slow = [elem.contains_point(p) for p in points]
    assert fast == slow


def test_parameters_at_xy_parity_graph(make_element, force_python_path):
    def make_surface():
        return make_element("0.1*(x**2) + 0.2*(y**2)", "-1").surface1

    targets = [(0.3, 0.2), (-0.7, 0.4), (0.0, 0.0), (1.1, -0.9)]
    fast_surface = make_surface()
    fast = [fast_surface.parameters_at_xy(x, y) for x, y in targets]
    slow_surface = make_surface()
    force_python_path([_SurfaceHolder(slow_surface)])
    slow = [slow_surface.parameters_at_xy(x, y) for x, y in targets]
    for (fu, fv), suv in zip(fast, slow):
        assert suv is not None
        assert fu == pytest.approx(suv[0], abs=1e-7)
        assert fv == pytest.approx(suv[1], abs=1e-7)


def test_parameters_at_xy_parity_non_graph(force_python_path):
    # A cylinder is not a graph over local XY, exercising the seed-grid branch.
    def make_surface():
        return Surface(
            "cos(u)", "v", "sin(u)", u_range=(0.0, 2 * np.pi), v_range=(-1.0, 1.0)
        )

    targets = [(1.0, 0.3), (0.0, 0.5), (-1.0, -0.2)]
    fast_surface = make_surface()
    fast = [fast_surface.parameters_at_xy(x, y) for x, y in targets]
    slow_surface = make_surface()
    force_python_path([_SurfaceHolder(slow_surface)])
    slow = [slow_surface.parameters_at_xy(x, y) for x, y in targets]
    for fu, suv in zip(fast, slow):
        if fu is None or suv is None:
            assert fu is None and suv is None
            continue
        assert fu[0] == pytest.approx(suv[0], abs=1e-7)
        assert fu[1] == pytest.approx(suv[1], abs=1e-7)
