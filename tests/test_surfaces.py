"""Surface geometry: sag, gradients, normals, intersections."""

import numpy as np
import pytest

from optics import Plane, ConicSurface, PolySagSurface
from optics.surfaces import PLANE, CONIC, POLY

H = 1e-4


def _fd(sag, x, y, axis):
    if axis == 0:
        return (sag(x + H, y) - sag(x - H, y)) / (2 * H)
    return (sag(x, y + H) - sag(x, y - H)) / (2 * H)


def test_plane_sag_constant():
    p = Plane("p", z0=3.0)
    x = np.array([[0.0, 5.0]]); y = np.array([[1.0, -2.0]])
    assert np.allclose(p.sag(x, y), 3.0)
    gx, gy = p.gradient(x, y)
    assert np.allclose(gx, 0.0) and np.allclose(gy, 0.0)


def test_plane_normal_z():
    p = Plane("p", z0=0.0)
    x = np.array([[1.0]]); y = np.array([[1.0]])
    n = p.normal(x, y)
    assert np.allclose(n, [[0.0, 0.0, 1.0]])


def test_plane_intersects_vertical_rays():
    p = Plane("p", z0=10.0)
    origins = np.array([[0.0, 0.0, 0.0]])
    dirs = np.array([[0.0, 0.0, 1.0]])
    t, pts = p.intersect(origins, dirs)
    assert t[0] == pytest.approx(10.0)
    assert pts[0, 2] == pytest.approx(10.0)


def test_plane_ray_parallel_to_plane_is_nan():
    p = Plane("p", z0=10.0)
    origins = np.array([[0.0, 0.0, 0.0]])
    dirs = np.array([[1.0, 0.0, 0.0]])
    t, _ = p.intersect(origins, dirs)
    assert np.isnan(t[0])


def test_conic_sag_vertex_matches_paraboloid():
    s = ConicSurface("c", radius=100.0, conicity=0.0, z0=0.0)
    # Paraxial sag ~ c*r^2/2 with c = 1/R; use small r so the approximation holds.
    x = np.array([[0.5, 1.0]]); y = np.array([[0.0, 0.5]])
    c = 1.0 / 100.0
    expected = c * (x ** 2 + y ** 2) / 2.0
    assert np.allclose(s.sag(x, y), expected, atol=1e-6)


def test_conic_gradient_matches_finite_difference():
    s = ConicSurface("c", radius=100.0, conicity=-0.5, z0=0.0)
    x = np.array([[10.0, 5.0]]); y = np.array([[5.0, 8.0]])
    gx, gy = s.gradient(x, y)
    assert np.allclose(gx, _fd(s.sag, x, y, 0), atol=1e-4)
    assert np.allclose(gy, _fd(s.sag, x, y, 1), atol=1e-4)


def test_conic_normal_is_unit_and_points_up():
    s = ConicSurface("c", radius=100.0, conicity=0.0, z0=0.0)
    x = np.array([[15.0]]); y = np.array([[10.0]])
    n = s.normal(x, y).reshape(-1, 3)[0]
    assert np.allclose(np.linalg.norm(n), 1.0)
    assert n[2] > 0.0


def test_conic_curvature_zero_for_infinite_radius():
    s = ConicSurface("c", radius=np.inf, conicity=0.0)
    assert s.curvature == 0.0
    x = np.array([[1.0]]); y = np.array([[1.0]])
    assert np.allclose(s.sag(x, y), 0.0)


def test_parabola_vs_sphere_vs_hyperbola_signs():
    # Same vertex radius, different conicity -> different edge sag.
    x = np.array([[20.0]]); y = np.array([[0.0]])
    R = 100.0
    para = ConicSurface("para", radius=R, conicity=-1.0).sag(x, y)[0, 0]
    sph = ConicSurface("sph", radius=R, conicity=0.0).sag(x, y)[0, 0]
    hyp = ConicSurface("hyp", radius=R, conicity=-2.0).sag(x, y)[0, 0]
    assert para > 0.0 and sph > 0.0 and hyp > 0.0
    # More negative conicity bows the surface out further at the edge:
    # sphere (k=0) < parabola (k=-1) < hyperbola (k=-2).
    assert sph < para < hyp


def test_poly_sag_gradient_matches_finite_difference():
    P = PolySagSurface("poly", order=4)
    P.coeffs[2, 0] = 0.01
    P.coeffs[0, 2] = -0.02
    x = np.array([[3.0, 1.0]]); y = np.array([[2.0, 4.0]])
    gx, gy = P.gradient(x, y)
    assert np.allclose(gx, _fd(P.sag, x, y, 0), atol=1e-3)
    assert np.allclose(gy, _fd(P.sag, x, y, 1), atol=1e-3)


def test_poly_bad_coeff_shape_raises():
    with pytest.raises(ValueError):
        PolySagSurface("poly", order=2, coeffs=np.zeros((3, 2)))


@pytest.mark.parametrize("kind,expected", [
    (Plane(), PLANE),
    (ConicSurface(), CONIC),
    (PolySagSurface(order=1), POLY),
])
def test_kind_codes(kind, expected):
    assert kind.kind == expected


def test_numba_descriptor_aperture_encoding():
    p = Plane("p", z0=2.0, aperture=5.0)
    kind, c0 = p.numba_descriptor()
    assert kind == PLANE
    assert c0.shape == (2,)
    assert c0[0] == pytest.approx(2.0)
    assert c0[1] == pytest.approx(25.0)  # aperture^2

    p_no = Plane("p", z0=2.0)
    _, c0 = p_no.numba_descriptor()
    assert c0[1] == pytest.approx(-1.0)  # no aperture

    c = ConicSurface("c", radius=100.0, conicity=0.5, z0=1.0, aperture=7.0)
    kind, c0 = c.numba_descriptor()
    assert kind == CONIC
    assert c0.shape == (4,)
    assert c0[3] == pytest.approx(49.0)
