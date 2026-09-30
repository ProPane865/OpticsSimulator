import numpy as np

import main
import visualize


def test_surface_positions_unit_plane(make_element):
    elem = make_element("0", "-2")
    verts, faces = visualize.surface_positions(elem, elem.surface1, n=16)
    assert verts.shape == (16 * 16, 3)
    np.testing.assert_allclose(verts[:, 2], 0.0, atol=1e-12)
    assert faces.shape[1] == 3
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))
    assert len(faces) == 2 * (16 - 1) * (16 - 1)


def test_surface_positions_sphere_hemisphere(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    verts, faces = visualize.surface_positions(elem, elem.surface1, n=64)
    assert len(verts) > 0
    assert np.all(np.isfinite(verts))
    np.testing.assert_allclose(np.linalg.norm(verts, axis=1), 1.0, atol=1e-9)
    assert np.all(verts[:, 2] >= -1e-12)
    assert len(faces) > 0
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))


def test_surface_positions_plane_below(make_element):
    elem = make_element("0", "-2")
    verts, faces = visualize.surface_positions(elem, elem.surface2, n=16)
    assert verts.shape == (16 * 16, 3)
    np.testing.assert_allclose(verts[:, 2], -2.0, atol=1e-12)


def test_surface_positions_tilted_matches_rotation(make_element):
    default = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    tilted = make_element("sqrt(1 - (x**2) - (y**2))", "0", orientation=np.array([0.0, 1.0, 0.0]))
    v0, f0 = visualize.surface_positions(default, default.surface1, n=32)
    v1, f1 = visualize.surface_positions(tilted, tilted.surface1, n=32)
    assert f0.shape == f1.shape
    np.testing.assert_allclose(v1, (tilted.r_matrix @ v0.T).T, atol=1e-12)


def test_trace_ray_on_axis(sphere):
    ray = main.Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]))
    p1, out = visualize.trace_ray(sphere, ray)
    assert p1 is not None
    assert np.all(np.isfinite(p1))
    assert np.isclose(np.linalg.norm(p1), 1.0, atol=1e-8)
    assert out is not None
    np.testing.assert_allclose(out.direction, [0.0, 0.0, -1.0], atol=1e-9)
    assert np.isclose(np.linalg.norm(out.origin), 1.0, atol=1e-8)


def test_trace_ray_off_axis_finite(sphere):
    ray = main.Ray(np.array([0.5, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]))
    p1, out = visualize.trace_ray(sphere, ray)
    assert p1 is not None
    assert np.all(np.isfinite(p1))
    assert out is not None
    assert np.all(np.isfinite(out.direction))
    assert np.all(np.isfinite(out.origin))


def test_trace_ray_miss_returns_none(sphere):
    ray = main.Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, 1.0]))
    p1, out = visualize.trace_ray(sphere, ray)
    assert p1 is None
    assert out is None


def _face_normals(verts, faces):
    tri = verts[np.asarray(faces)]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return np.cross(e1, e2)


def test_surface_winding_consistent_per_surface(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "-sqrt(1 - (x**2) - (y**2))")

    v1, f1 = visualize.surface_positions(elem, elem.surface1, n=32)
    assert f1.shape[0] > 0
    n1 = _face_normals(v1, f1)
    assert np.all(n1[:, 2] > 0), "upper hemisphere normals must all face +z (outward)"

    v2, f2 = visualize.surface_positions(elem, elem.surface2, n=32)
    assert f2.shape[0] > 0
    n2 = _face_normals(v2, f2)
    assert np.all(n2[:, 2] < 0), "lower hemisphere normals must all face -z (outward)"
