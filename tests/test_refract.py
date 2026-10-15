import numpy as np
import pytest

from optics.primitive.hit import MeshHit
from optics.primitive.ray import Ray
from optics.refraction import (
    refract_or_reflect,
    trace_direction,
    trace_stack,
    trace_stack_path,
)

BICONE_S1 = "3 - 0.3*sqrt(x**2 + y**2)"
BICONE_S2 = "-1 + 2*sqrt(x**2 + y**2)"


def _refract_direction(d_in: np.ndarray, n_face: np.ndarray, eta: float) -> np.ndarray:
    """Standard vector refraction, as in old/main_old.py.

    n_face: unit surface normal opposing the incident ray (n_face @ d_in < 0).
    eta: refractive index of the incident medium over that of the transmitted one.
    """
    cos_i = -float(np.dot(n_face, d_in))
    k = 1.0 - eta**2 * (1.0 - cos_i**2)
    if k < 0:
        return d_in - 2.0 * float(np.dot(n_face, d_in)) * n_face
    cos_t = np.sqrt(k)
    t = eta * d_in + (eta * cos_i - cos_t) * n_face
    return t / np.linalg.norm(t)


def test_refract_or_reflect_returns_unit_direction():
    incident = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    normal = np.array([0.0, 0.0, -1.0])

    d, refracted = refract_or_reflect(incident, normal, 1.0, 1.5)
    assert refracted
    np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-12)

    d, refracted = refract_or_reflect(incident, normal, 1.5, 1.0)
    assert not refracted
    np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-12)


def test_sphere_on_axis_no_deflection(sphere):
    ray = Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0)
    out = sphere.trace(ray).outgoing_ray
    np.testing.assert_allclose(out.direction, [0.0, 0.0, -1.0], atol=1e-9)
    assert np.isclose(np.linalg.norm(out.origin), 1.0, atol=1e-8)


def test_slab_exit_parallel_to_incident(make_element):
    elem = make_element("0", "-2", n=1.5)
    d = np.array([0.4, -0.2, -1.0])
    ray = Ray(np.array([0.5, 0.0, 2.5]), d, 1.0)
    out = elem.trace(ray).outgoing_ray
    np.testing.assert_allclose(out.direction, d / np.linalg.norm(d), atol=1e-9)
    assert np.isclose(out.origin[2], -2.0, atol=1e-8)


def test_slab_lateral_shift_matches_analytic(make_element):
    elem = make_element("0", "-2", n=1.5)
    d = np.array([0.4, -0.2, -1.0])
    d = d / np.linalg.norm(d)
    ray = Ray(np.array([0.5, 0.0, 2.5]), d, 1.0)
    out = elem.trace(ray).outgoing_ray

    t = (2.5 - (-2.0)) / (-d[2])
    straight = ray.at(t)
    shift = np.linalg.norm(np.cross(out.origin - straight, out.direction))

    sin_i = float(np.linalg.norm(d[:2]))
    cos_i = float(-d[2])
    sin_t = sin_i / 1.5
    expected = 2.0 * (sin_i - cos_i * np.tan(np.arcsin(sin_t)))
    assert np.isclose(shift, expected, atol=1e-3)


def test_sphere_off_axis_matches_reference(sphere):
    ray = Ray(np.array([0.5, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 1.0)
    out = sphere.trace(ray).outgoing_ray

    hit1 = sphere.surface1.intersect(ray)
    p1, u1, v1 = hit1.point, hit1.u, hit1.v
    n1 = -sphere.surface1.normal(u1, v1)
    d1 = _refract_direction(ray.direction, n1, 1.0 / sphere.material.refractive_index)

    hit2 = sphere.surface2.intersect(Ray(p1, d1, 1.0))
    p2, u2, v2 = hit2.point, hit2.u, hit2.v
    n2 = -sphere.surface2.normal(u2, v2)
    d2 = _refract_direction(d1, n2, sphere.material.refractive_index)

    np.testing.assert_allclose(out.direction, d2, atol=1e-9)
    np.testing.assert_allclose(out.origin, p2, atol=1e-8)


def test_tilted_sphere_on_axis_no_deflection(make_element):
    s = np.array([0.0, 1.0, 0.0])
    elem = make_element(
        "sqrt(1 - (x**2) - (y**2))",
        "-sqrt(1 - (x**2) - (y**2))",
        orientation=s,
    )
    ray = Ray(2.0 * s, -s, 1.0)
    out = elem.trace(ray).outgoing_ray
    np.testing.assert_allclose(out.direction, -s, atol=1e-9)
    assert np.isclose(np.linalg.norm(out.origin), 1.0, atol=1e-8)


def test_unit_index_plane_slab_is_straight(make_element):
    elem = make_element("0", "-2", n=1.0)
    ray = Ray(np.array([1.0, 0.0, 4.0]), np.array([1.0, 0.0, -2.0]), 1.0)
    out = elem.trace(ray).outgoing_ray
    np.testing.assert_allclose(out.direction, ray.direction, atol=1e-9)
    t = (4.0 - (-2.0)) / (-ray.direction[2])
    expected = ray.at(t)
    np.testing.assert_allclose(out.origin, expected, atol=1e-8)


def test_bicone_launch_is_supercritical(make_element):
    elem = make_element(BICONE_S1, BICONE_S2, n=1.5)
    ray = Ray(np.array([0.05, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    hit1 = elem.surface1.intersect(ray)
    p1, u1, v1 = hit1.point, hit1.u, hit1.v
    n1 = -elem.surface1.normal(u1, v1)
    d1 = _refract_direction(ray.direction, n1, 1.0 / 1.5)

    hit2 = elem.surface2.intersect(Ray(p1, d1, 1.0))
    u2, v2 = hit2.u, hit2.v
    n2 = -elem.surface2.normal(u2, v2)
    cos_i2 = -float(np.dot(n2, d1))

    assert cos_i2 < 1.0 / 1.5
    assert 1.0 - 1.5**2 * (1.0 - cos_i2**2) < 0.0


def test_total_internal_reflection_reflects_inside_element(make_element):
    elem = make_element(BICONE_S1, BICONE_S2, n=1.5)
    ray = Ray(np.array([0.05, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    hit1 = elem.surface1.intersect(ray)
    p1, u1, v1 = hit1.point, hit1.u, hit1.v
    n1 = -elem.surface1.normal(u1, v1)
    d1 = _refract_direction(ray.direction, n1, 1.0 / 1.5)

    hit2 = elem.surface2.intersect(Ray(p1, d1, 1.0))
    u2, v2 = hit2.u, hit2.v
    n2 = -elem.surface2.normal(u2, v2)
    expected = d1 - 2.0 * float(np.dot(n2, d1)) * n2

    result = elem.trace(ray)
    assert len(result.internal_path) >= 2
    np.testing.assert_allclose(
        result.internal_path[1].direction, expected, atol=1e-9
    )
    assert result.exit_hit is not None
    assert result.outgoing_ray is not None


def test_grazing_ray_does_not_crash(sphere):
    ray = Ray(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 1.0)
    result = sphere.trace(ray)
    out = result.outgoing_ray
    assert out is None or np.all(np.isfinite(out.direction))


def test_refract_direction_on_axis_no_deflection():
    d = np.array([0.0, 0.0, -1.0])
    n = np.array([0.0, 0.0, -1.0])
    out = trace_direction(d, n, n_from=1.0, n_to=1.5)
    np.testing.assert_allclose(out, d, atol=1e-12)


def test_refract_direction_preserves_unit_length():
    d = np.array([0.6, 0.0, -0.8])
    n = np.array([0.0, 0.0, -1.0])
    out = trace_direction(d, n, n_from=1.0, n_to=1.5)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-12)


def test_refract_direction_satisfies_snells_law():
    d = np.array([0.6, 0.0, -0.8])
    n = np.array([0.0, 0.0, -1.0])
    out = trace_direction(d, n, n_from=1.0, n_to=1.5)
    sin_i = float(np.linalg.norm(d[:2]))
    sin_t = float(np.linalg.norm(out[:2]))
    np.testing.assert_allclose(sin_t, (1.0 / 1.5) * sin_i, atol=1e-12)


def test_refract_direction_total_internal_reflection_reflects():
    d = np.array([np.sqrt(3.0) / 2.0, 0.0, -0.5])
    n = np.array([0.0, 0.0, -1.0])
    out = trace_direction(d, n, n_from=1.5, n_to=1.0)
    expected = d - 2.0 * float(np.dot(n, d)) * n
    np.testing.assert_allclose(out, expected, atol=1e-12)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-12)


def test_refract_direction_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d)
        n = rng.normal(size=3)
        n = n / np.linalg.norm(n)
        if float(np.dot(n, d)) < 0.0:
            n = -n
        n_from = 1.0 + rng.random()
        n_to = 1.0 + rng.random()
        out = trace_direction(d, n, n_from=n_from, n_to=n_to)
        expected = _refract_direction(d, -n, n_from / n_to)
        np.testing.assert_allclose(out, expected, atol=1e-12)

def test_trace_miss_returns_empty_trace(make_element):
    elem = make_element("0", "-2")

    ray = Ray(
        np.array([15.0, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
        1.0,
    )

    result = elem.trace(ray)

    assert result.incident_hit is None
    assert result.exit_hit is None
    assert result.incident_ray is ray
    assert result.internal_path == ()
    assert result.outgoing_ray is None


def test_hit_surface1_only_returns_incident_hit_internal_path(make_element):
    elem = make_element("0", "-2")

    ray = Ray(
        np.array([9.0, 0.0, 2.0]),
        np.array([0.5, 0.0, -1.0]),
        1.0,
    )

    result = elem.trace(ray)

    assert result.incident_hit is not None
    assert result.exit_hit is None
    assert result.incident_ray is ray
    assert len(result.internal_path) == 1
    assert result.outgoing_ray is None


def test_trace_conserves_ray_power(sphere):
    ray = Ray(np.array([0.5, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]), 3.0)
    result = sphere.trace(ray)
    assert result.internal_path[0].power == pytest.approx(3.0)
    assert result.outgoing_ray.power == pytest.approx(3.0)


def test_trace_stack_independent_of_element_order(make_element):
    lens = make_element("sqrt(1 - (x**2) - (y**2))", "-1")
    plate = make_element("-4", "-5")
    ray = Ray(np.array([0.5, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    forward = trace_stack(ray, [lens, plate])
    reversed_ = trace_stack(ray, [plate, lens])

    assert forward is not None
    assert reversed_ is not None
    np.testing.assert_allclose(forward.direction, reversed_.direction, atol=1e-12)
    np.testing.assert_allclose(forward.origin, reversed_.origin, atol=1e-12)


def test_trace_stack_path_segment_kinds(make_element):
    lens = make_element("sqrt(1 - (x**2) - (y**2))", "-1")
    plate = make_element("-4", "-5")
    ray = Ray(np.array([0.5, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    segments, final = trace_stack_path(ray, [plate, lens], ray_length=2.0)
    kinds = [kind for _, _, kind in segments]

    assert final is not None
    assert kinds[0] == "incident"
    assert "internal" in kinds
    assert kinds[-1] == "exit"


def test_first_hit_prefers_sidewall(make_element):
    elem = make_element("0", "-2", aperture_radius=1.0)
    ray = Ray(np.array([2.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0]), 1.0)

    entry = elem.first_hit(ray)

    assert entry is not None
    assert isinstance(entry[1], MeshHit)
    np.testing.assert_allclose(entry[1].point, [1.0, 0.0, -1.0], atol=1e-9)


def test_sidewall_entry_refracts(make_element):
    elem = make_element("0", "-2", n=1.5, aperture_radius=1.0)
    ray = Ray(np.array([2.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0]), 1.0)

    result = elem.trace(ray)

    assert isinstance(result.incident_hit, MeshHit)
    assert result.outgoing_ray is not None
    np.testing.assert_allclose(
        result.outgoing_ray.direction, [-1.0, 0.0, 0.0], atol=1e-9
    )
    np.testing.assert_allclose(
        result.outgoing_ray.origin, [-1.0, 0.0, -1.0], atol=0.05
    )


def test_sidewall_entry_bends_toward_normal(make_element):
    elem = make_element("0", "-2", n=1.5, aperture_radius=1.0)
    d = np.array([-1.0, 0.0, -0.5])
    d = d / np.linalg.norm(d)
    ray = Ray(np.array([2.0, 0.0, -1.0]), d, 1.0)

    result = elem.trace(ray)

    assert isinstance(result.incident_hit, MeshHit)
    internal = result.internal_path[0].direction
    assert abs(internal[2]) < abs(d[2])
    assert result.outgoing_ray is not None