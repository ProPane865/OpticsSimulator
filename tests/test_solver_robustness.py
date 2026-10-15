import numpy as np
import pytest

from optics.measurement_plane import MeasurementPlane
from optics.numerical.newton import _CONVERGED, _STALLED, _newton
from optics.primitive import CircularAperture, Ray, Surface

GRAZING_ORIGINS = [
    [0.0, 0.0, 0.0],
    [0.3, 0.2, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [-3.0, 0.0, 0.999],
]
GRAZING_DIRECTIONS = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.01],
    [0.6, 0.0, 0.8],
    [0.0, 0.0, -1.0],
]


def _snell_exit(d_in: np.ndarray, n_from: float, n_to: float) -> np.ndarray:
    """Exit direction for ``d_in`` crossing a plane with normal +-z, graph form."""
    d = d_in / np.linalg.norm(d_in)
    sin_i = float(np.hypot(d[0], d[1]))
    sin_t = n_from / n_to * sin_i
    assert sin_t <= 1.0, "expected a subcritical hit"
    transverse = d[:2] / sin_i if sin_i > 0.0 else np.zeros(2)
    z = np.sign(d[2]) * np.sqrt(1.0 - sin_t**2)
    return np.array([transverse[0] * sin_t, transverse[1] * sin_t, z])


def test_newton_converges_on_linear_system():
    result = _newton(lambda x: np.array([x[0] - 1.0]), lambda x: np.array([[1.0]]), (0.0,))
    assert result is not None
    x, f_norm, status = result
    assert status == _CONVERGED
    assert f_norm < 1e-12
    np.testing.assert_allclose(x[0], 1.0, atol=1e-12)


def test_newton_aborts_on_non_finite_initial_residual():
    assert _newton(lambda x: np.array([np.nan]), lambda x: np.array([[1.0]]), (0.0,)) is None


def test_newton_reports_stalled_when_steps_underflow():
    result = _newton(lambda x: np.array([1e-6]), lambda x: np.array([[1e12]]), (0.0,))
    assert result is not None
    _, _, status = result
    assert status == _STALLED


def test_newton_backtracks_when_step_leaves_domain():
    """A step landing where the residual is undefined is halved, not abandoned."""

    def residual(x):
        if x[0] > 1.0:
            return None
        return np.array([x[0] - 1.0])

    result = _newton(residual, lambda x: np.array([[0.5]]), (0.0,))
    assert result is not None
    x, _, status = result
    assert status == _CONVERGED
    np.testing.assert_allclose(x[0], 1.0, atol=1e-12)


def test_newton_uses_least_squares_for_rank_deficient_jacobian():
    def residual(x):
        return np.array([x[0] + x[1] - 1.0, 2.0 * x[0] + 2.0 * x[1] - 2.0])

    result = _newton(residual, lambda x: np.array([[1.0, 1.0], [2.0, 2.0]]), (0.0, 0.0))
    assert result is not None
    x, f_norm, _ = result
    assert f_norm < 1e-11
    np.testing.assert_allclose(x[0] + x[1], 1.0, atol=1e-9)


def test_normal_at_aperture_rim_is_finite_and_radial():
    surface = Surface("u", "v", "sqrt(1 - u**2 - v**2)", aperture=CircularAperture(1.0))
    normal = surface.normal(1.0, 0.0)
    assert normal is not None
    assert np.all(np.isfinite(normal))
    np.testing.assert_allclose(abs(normal[0]), 1.0, atol=1e-3)
    assert abs(normal[2]) < 1e-2


def test_contains_point_slab(make_element):
    elem = make_element("2", "0", n=1.5)
    assert elem.contains_point([0.0, 0.0, 1.0])
    assert not elem.contains_point([0.0, 0.0, 3.0])
    assert not elem.contains_point([0.0, 0.0, -1.0])


def test_contains_point_respects_aperture(make_element):
    elem = make_element("2", "0", n=1.5, aperture_radius=1.0)
    assert elem.contains_point([0.0, 0.0, 1.0])
    assert not elem.contains_point([2.0, 0.0, 1.0])


def test_contains_point_uses_element_frame(make_element):
    elem = make_element("2", "0", n=1.5, orientation=[1.0, 0.0, 0.0])
    assert elem.contains_point([1.0, 0.0, 0.0])
    assert not elem.contains_point([0.0, 3.0, 0.0])


def test_ray_starting_inside_slab_refracts_out(make_element):
    elem = make_element("2", "0", n=1.5)
    d = np.array([0.4, -0.2, -1.0])
    d = d / np.linalg.norm(d)
    result = elem.trace(Ray(np.array([0.5, 0.0, 1.0]), d, 1.0))

    out = result.outgoing_ray
    assert out is not None
    np.testing.assert_allclose(out.direction, _snell_exit(d, 1.5, 1.0), atol=1e-9)
    assert result.incident_hit is None
    assert result.exit_hit is not None
    np.testing.assert_allclose(result.exit_hit.point[2], 0.0, atol=1e-8)
    assert out.power == pytest.approx(1.0)


def test_ray_starting_inside_slab_exits_through_other_face(make_element):
    elem = make_element("2", "0", n=1.5)
    d = np.array([0.4, -0.2, 1.0])
    d = d / np.linalg.norm(d)
    result = elem.trace(Ray(np.array([0.5, 0.0, 1.0]), d, 1.0))

    assert result.exit_hit is not None
    np.testing.assert_allclose(result.exit_hit.point[2], 2.0, atol=1e-8)
    np.testing.assert_allclose(
        result.outgoing_ray.direction, _snell_exit(d, 1.5, 1.0), atol=1e-9
    )


def test_ray_starting_inside_reports_internal_path(make_element):
    elem = make_element("2", "0", n=1.5)
    ray = Ray(np.array([0.5, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]), 1.0)
    result = elem.trace(ray)

    assert result.internal_path
    np.testing.assert_allclose(result.internal_path[0].origin, ray.origin, atol=1e-12)


def test_radial_ray_inside_sphere_exits_undeviated(sphere):
    result = sphere.trace(Ray(np.zeros(3), np.array([0.0, 0.0, 1.0]), 1.0))
    out = result.outgoing_ray
    assert out is not None
    np.testing.assert_allclose(out.origin, [0.0, 0.0, 1.0], atol=1e-8)
    np.testing.assert_allclose(out.direction, [0.0, 0.0, 1.0], atol=1e-9)


@pytest.mark.parametrize("origin", GRAZING_ORIGINS)
@pytest.mark.parametrize("direction", GRAZING_DIRECTIONS)
def test_grazing_rays_never_raise(sphere, origin, direction):
    ray = Ray(np.asarray(origin, dtype=float), np.asarray(direction, dtype=float), 1.0)
    result = sphere.trace(ray)

    for segment in result.internal_path:
        assert np.all(np.isfinite(segment.origin))
        assert np.all(np.isfinite(segment.direction))
    out = result.outgoing_ray
    assert out is None or np.all(np.isfinite(out.direction))


def test_plane_inside_element_measures_internal_ray(make_element):
    elem = make_element("2", "0", n=1.5)
    plane = MeasurementPlane(
        origin=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=(-2.0, 2.0),
        v_range=(-2.0, 2.0),
        resolution=(16, 16),
    )
    ray = Ray(np.array([0.5, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)

    total = plane.measure([ray], [elem])
    assert total == pytest.approx(1.0)

    i, j = plane.discretize(0.5, 0.0)
    assert plane.grid[j, i] == pytest.approx(1.0)
    assert float(np.sum(plane.grid)) == pytest.approx(1.0)
