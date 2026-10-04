import numpy as np
import pytest

import main

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


def test_sphere_on_axis_no_deflection(sphere):
    ray = main.Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]))
    out = sphere.refract(ray)
    np.testing.assert_allclose(out.direction, [0.0, 0.0, -1.0], atol=1e-9)
    assert np.isclose(np.linalg.norm(out.origin), 1.0, atol=1e-8)


def test_slab_exit_parallel_to_incident(make_element):
    elem = make_element("0", "-2", n=1.5)
    d = np.array([0.4, -0.2, -1.0])
    ray = main.Ray(np.array([0.5, 0.0, 2.5]), d)
    out = elem.refract(ray)
    np.testing.assert_allclose(out.direction, d / np.linalg.norm(d), atol=1e-9)
    assert np.isclose(out.origin[2], -2.0, atol=1e-8)


def test_slab_lateral_shift_matches_analytic(make_element):
    elem = make_element("0", "-2", n=1.5)
    d = np.array([0.4, -0.2, -1.0])
    d = d / np.linalg.norm(d)
    ray = main.Ray(np.array([0.5, 0.0, 2.5]), d)
    out = elem.refract(ray)

    t = (2.5 - (-2.0)) / (-d[2])
    straight = ray.origin + t * d
    shift = np.linalg.norm(np.cross(out.origin - straight, out.direction))

    sin_i = float(np.linalg.norm(d[:2]))
    cos_i = float(-d[2])
    sin_t = sin_i / 1.5
    expected = 2.0 * (sin_i - cos_i * np.tan(np.arcsin(sin_t)))
    assert np.isclose(shift, expected, atol=1e-3)


def test_sphere_off_axis_matches_reference(sphere):
    ray = main.Ray(np.array([0.5, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]))
    out = sphere.refract(ray)

    hit1 = sphere.surface1.intersect(ray)
    p1, u1, v1 = hit1
    n1 = -sphere.surface1.normal(u1, v1)
    d1 = _refract_direction(ray.direction, n1, 1.0 / sphere.n)

    hit2 = sphere.surface2.intersect(main.Ray(p1, d1))
    p2, u2, v2 = hit2
    n2 = -sphere.surface2.normal(u2, v2)
    d2 = _refract_direction(d1, n2, sphere.n)

    np.testing.assert_allclose(out.direction, d2, atol=1e-9)
    np.testing.assert_allclose(out.origin, p2, atol=1e-8)


def test_tilted_sphere_on_axis_no_deflection(make_element):
    s = np.array([0.0, 1.0, 0.0])
    elem = make_element(
        "sqrt(1 - (x**2) - (y**2))",
        "-sqrt(1 - (x**2) - (y**2))",
        orientation=s,
    )
    ray = main.Ray(2.0 * s, -s)
    out = elem.refract(ray)
    np.testing.assert_allclose(out.direction, -s, atol=1e-9)
    assert np.isclose(np.linalg.norm(out.origin), 1.0, atol=1e-8)


def test_unit_index_plane_slab_is_straight(make_element):
    elem = make_element("0", "-2", n=1.0)
    ray = main.Ray(np.array([1.0, 0.0, 4.0]), np.array([1.0, 0.0, -2.0]))
    out = elem.refract(ray)
    np.testing.assert_allclose(out.direction, ray.direction, atol=1e-9)
    t = (4.0 - (-2.0)) / (-ray.direction[2])
    expected = ray.origin + t * ray.direction
    np.testing.assert_allclose(out.origin, expected, atol=1e-8)


def test_bicone_launch_is_supercritical(make_element):
    elem = make_element(BICONE_S1, BICONE_S2, n=1.5)
    ray = main.Ray(np.array([0.05, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]))

    hit1 = elem.surface1.intersect(ray)
    p1, u1, v1 = hit1
    n1 = -elem.surface1.normal(u1, v1)
    d1 = _refract_direction(ray.direction, n1, 1.0 / 1.5)

    hit2 = elem.surface2.intersect(main.Ray(p1, d1))
    p2, u2, v2 = hit2
    n2 = -elem.surface2.normal(u2, v2)
    cos_i2 = -float(np.dot(n2, d1))

    assert cos_i2 < 1.0 / 1.5
    assert 1.0 - 1.5**2 * (1.0 - cos_i2**2) < 0.0


@pytest.mark.xfail(
    strict=True,
    reason="TIR currently propagates an all-NaN direction; expected behavior is the specular reflection",
)
def test_total_internal_reflection_returns_reflected_ray(make_element):
    elem = make_element(BICONE_S1, BICONE_S2, n=1.5)
    ray = main.Ray(np.array([0.05, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]))

    hit1 = elem.surface1.intersect(ray)
    p1, u1, v1 = hit1
    n1 = -elem.surface1.normal(u1, v1)
    d1 = _refract_direction(ray.direction, n1, 1.0 / 1.5)

    hit2 = elem.surface2.intersect(main.Ray(p1, d1))
    p2, u2, v2 = hit2
    n2 = -elem.surface2.normal(u2, v2)
    expected = d1 - 2.0 * float(np.dot(n2, d1)) * n2

    out = elem.refract(ray)
    np.testing.assert_allclose(out.direction, expected, atol=1e-9)


@pytest.mark.xfail(
    strict=True,
    reason="grazing entry makes the intersection/normal undefined and refract raises TypeError",
)
def test_grazing_ray_does_not_crash(sphere):
    ray = main.Ray(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    out = sphere.refract(ray)
    assert np.all(np.isfinite(out.direction))
