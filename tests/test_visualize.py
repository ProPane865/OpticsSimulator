import types
import numpy as np
import rendering.visualize as visualize

from rendering.meshing import SurfaceMesher
from optics.ray import Ray
from optics.surface import Surface

def test_surface_positions_unit_plane(make_element):
    elem = make_element("0", "-2")
    verts, faces = visualize.surface_positions(elem.surface1, n=16)
    assert verts.shape == (16 * 16, 3)
    np.testing.assert_allclose(verts[:, 2], 0.0, atol=1e-12)
    assert faces.shape[1] == 3
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))
    assert len(faces) == 2 * (16 - 1) * (16 - 1)


def test_surface_positions_sphere_hemisphere(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    verts, faces = visualize.surface_positions(elem.surface1, n=64)
    assert len(verts) > 0
    assert np.all(np.isfinite(verts))
    np.testing.assert_allclose(np.linalg.norm(verts, axis=1), 1.0, atol=1e-9)
    assert np.all(verts[:, 2] >= -1e-12)
    assert len(faces) > 0
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))


def test_surface_positions_plane_below(make_element):
    elem = make_element("0", "-2")
    verts, faces = visualize.surface_positions(elem.surface2, n=16)
    assert verts.shape == (16 * 16, 3)
    np.testing.assert_allclose(verts[:, 2], -2.0, atol=1e-12)


def test_surface_positions_tilted_matches_rotation(make_element):
    default = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    tilted = make_element("sqrt(1 - (x**2) - (y**2))", "0", orientation=np.array([0.0, 1.0, 0.0]))
    v0, f0 = visualize.surface_positions(default.surface1, n=32)
    v1, f1 = visualize.surface_positions(tilted.surface1, n=32)
    assert f0.shape == f1.shape
    np.testing.assert_allclose(v1, tilted.transform.points_to_world(v0), atol=1e-12)


def test_trace_ray_on_axis(sphere):
    ray = Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]))
    trace = sphere.trace(ray)
    p1 = trace.incident_hit.point
    out = trace.outgoing_ray
    assert p1 is not None
    assert np.all(np.isfinite(p1))
    assert np.isclose(np.linalg.norm(p1), 1.0, atol=1e-8)
    assert out is not None
    np.testing.assert_allclose(out.direction, [0.0, 0.0, -1.0], atol=1e-9)
    assert np.isclose(np.linalg.norm(out.origin), 1.0, atol=1e-8)


def test_trace_ray_off_axis_finite(sphere):
    ray = Ray(np.array([0.5, 0.0, 2.5]), np.array([0.0, 0.0, -1.0]))
    trace = sphere.trace(ray)
    p1 = trace.incident_hit.point
    out = trace.outgoing_ray
    assert p1 is not None
    assert np.all(np.isfinite(p1))
    assert out is not None
    assert np.all(np.isfinite(out.direction))
    assert np.all(np.isfinite(out.origin))


def test_trace_ray_miss_returns_none(sphere):
    ray = Ray(np.array([0.0, 0.0, 2.5]), np.array([0.0, 0.0, 1.0]))
    trace = sphere.trace(ray)
    assert trace.incident_hit is None
    assert trace.outgoing_ray is None


def _face_normals(verts, faces):
    tri = verts[np.asarray(faces)]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return np.cross(e1, e2)


def test_surface_winding_consistent_per_surface(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "-sqrt(1 - (x**2) - (y**2))")

    v1, f1 = visualize.surface_positions(elem.surface1, n=32)
    assert f1.shape[0] > 0
    n1 = _face_normals(v1, f1)
    assert np.all(n1[:, 2] > 0), "upper hemisphere normals must all face +z (outward)"

    v2, f2 = visualize.surface_positions(elem.surface2, n=32)
    assert f2.shape[0] > 0
    n2 = _face_normals(v2, f2)
    assert np.all(n2[:, 2] < 0), "lower hemisphere normals must all face -z (outward)"


def test_surface_mesh_reaches_domain_boundary_sphere(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    verts, faces = visualize.surface_positions(elem.surface1, n=128)
    assert len(verts) > 0
    assert np.all(np.isfinite(verts))
    r = np.linalg.norm(verts[:, :2], axis=1)
    assert r.max() > 0.9999, "mesh must reach the true domain boundary at the vertical rim"
    assert np.all(r <= 1.0 + 1e-9)
    assert len(faces) > 0
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))


def test_ray_segment_rendered_above_surfaces():
    line = visualize._make_segment(
        np.array([0.0, 0.0, 3.0]),
        np.array([0.0, 0.0, 0.5]),
        (1.0, 0.0, 0.0, 1.0),
    )
    assert line is not None
    subvisual = line._subvisuals[0]
    assert subvisual._vshare.gl_state["preset"] == "translucent"
    assert subvisual._vshare.gl_state["depth_test"] is False


def test_ray_marker_rendered_above_surfaces():
    marker = visualize._make_marker(np.array([0.0, 0.0, 1.0]), (1.0, 0.0, 0.0, 1.0))
    assert marker._vshare.gl_state["depth_test"] is False


def test_make_segment_rejects_non_finite_points():
    assert visualize._make_segment(np.array([np.nan, 0.0, 0.0]), np.zeros(3), (1.0, 0.0, 0.0, 1.0)) is None


def test_surface_mesh_reaches_vertical_wall_boundary():
    s = Surface("sqrt(1 - v**2)", "u", "v", u_range=(-10.0, 10.0), v_range=(-10.0, 10.0))
    verts, faces = SurfaceMesher(s).mesh(128)
    assert len(verts) > 0
    assert np.all(np.isfinite(verts))
    assert len(faces) > 0
    assert np.all(faces >= 0)
    assert np.all(faces < len(verts))
    assert verts[:, 2].max() > 0.9999
    assert verts[:, 2].min() < -0.9999


def test_make_wall_visual(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "-sqrt(1 - (x**2) - (y**2))", aperture_radius=0.5)
    wall = elem.sidewall
    assert wall is not None
    mesh = visualize.make_wall_visual(wall)
    assert mesh is not None
    assert mesh._vshare.gl_state["blend"] is True
    assert mesh._vshare.gl_state["depth_mask"] is True


def test_visualize_attaches_sidewall_when_aperture(make_element, monkeypatch):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "-sqrt(1 - (x**2) - (y**2))", aperture_radius=0.5)
    assert elem.sidewall is not None

    attached = []

    class _FakeView:
        def add(self, visual):
            attached.append(("view", visual))

    class _FakeCanvas:
        def __init__(self, **kwargs):
            self.central_widget = _FakeCentralWidget()

        def show(self, run=False):
            pass

    class _FakeCentralWidget:
        def add_view(self, **kwargs):
            return _FakeView()

    monkeypatch.setattr(visualize, "scene", types.SimpleNamespace(SceneCanvas=_FakeCanvas))
    monkeypatch.setattr(visualize, "make_surface_visual", lambda surface, color: ("surface", surface))
    monkeypatch.setattr(visualize, "make_wall_visual", lambda wall, color: ("wall", wall))
    monkeypatch.setattr(visualize, "draw_ray", lambda view, element, ray, ray_length=3.0: None)

    canvas = visualize.visualize(elem, [], show=False)

    kinds = [payload[0] for _, payload in attached if isinstance(payload, tuple)]
    assert kinds.count("surface") == 2
    assert "wall" in kinds
    assert canvas is not None
