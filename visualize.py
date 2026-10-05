from pathlib import Path

import numpy as np
from vispy import scene
from vispy.scene import visuals

from rendering.meshing import SurfaceMesher

SURFACE1_COLOR = (0.35, 0.65, 1.0, 0.25)
SURFACE2_COLOR = (0.2, 0.45, 0.9, 0.25)
SIDEWALL_COLOR = (0.55, 0.55, 0.65, 0.3)
INCIDENT_COLOR = (0.9, 0.2, 0.2, 1.0)
INTERNAL_COLOR = (1.0, 0.6, 0.1, 1.0)
EXIT_COLOR = (0.2, 0.75, 0.3, 1.0)
AXIS_COLORS = {
    "x": (0.85, 0.15, 0.15, 1.0),
    "y": (0.15, 0.65, 0.2, 1.0),
    "z": (0.15, 0.3, 0.85, 1.0),
}


def surface_positions(surface, n=128):
    return SurfaceMesher(surface).mesh(n)


def make_surface_visual(surface, color=SURFACE1_COLOR, **mesh_kwargs):
    n = mesh_kwargs.pop("n", 128)
    verts, faces = surface_positions(surface, n=n)
    kwargs = dict(
        color=color,
        shading="smooth",
    )
    kwargs.update(mesh_kwargs)

    mesh =  visuals.Mesh(verts, faces, **kwargs)
    mesh.set_gl_state(
        blend=True,
        depth_test=True,
        depth_mask=False,
        blend_func=("src_alpha", "one_minus_src_alpha")
    )
    
    return mesh


def make_wall_visual(sidewall, color=SIDEWALL_COLOR):
    verts, faces = sidewall.mesh()

    mesh = visuals.Mesh(verts, faces, color=color, shading="smooth")
    mesh.set_gl_state(
        blend=True,
        depth_test=True,
        depth_mask=False,
        blend_func=("src_alpha", "one_minus_src_alpha")
    )

    return mesh


def _marker_positions(pos, radius=0.03, n_phi=16, n_theta=8):
    v = np.linspace(0.0, np.pi, n_theta)
    u = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    V, U = np.meshgrid(v, u)
    x = radius * np.sin(V) * np.cos(U)
    y = radius * np.sin(V) * np.sin(U)
    z = radius * np.cos(V)
    verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()]) + np.asarray(pos, dtype=float)

    i = np.repeat(np.arange(n_theta - 1), n_phi)
    j = np.tile(np.arange(n_phi), n_theta - 1)
    a = i * n_phi + j
    b = (i + 1) * n_phi + j
    c = (i + 1) * n_phi + (j + 1) % n_phi
    d = i * n_phi + (j + 1) % n_phi
    faces = np.vstack([
        np.column_stack([a, b, c]),
        np.column_stack([a, c, d]),
    ])
    return verts, faces


def trace_ray(element, ray):
    hit = element.surface1.intersect(ray)
    p1 = hit[0] if hit is not None else None
    if p1 is not None and not np.all(np.isfinite(p1)):
        p1 = None
    if p1 is None:
        return None, None
    try:
        out = element.refract(ray)
    except (TypeError, ValueError):
        return p1, None
    return p1, out


def _attach(view, visual):
    view.add(visual)
    return visual


def _make_marker(pos, color, radius=0.03):
    verts, faces = _marker_positions(pos, radius=radius)
    marker = visuals.Mesh(verts, faces, color=color, shading="flat")
    marker.update_gl_state(depth_test=False)
    return marker


def _add_marker(view, pos, color, radius=0.03):
    return _attach(view, _make_marker(pos, color, radius=radius))


def _make_segment(a, b, color, width=3):
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    segment = visuals.Line(pos=np.array([a, b]), color=color, width=width)
    segment.update_gl_state(depth_test=False)
    return segment


def _add_segment(view, a, b, color, width=3):
    segment = _make_segment(a, b, color, width=width)
    if segment is None:
        return None
    return _attach(view, segment)


def _add_axes(view, length=2.0, width=2):
    origin = np.zeros(3)
    for name, ix in (("x", 0), ("y", 1), ("z", 2)):
        end = np.zeros(3)
        end[ix] = length
        _add_segment(view, origin, end, AXIS_COLORS[name], width=width)


def draw_ray(view, element, ray, ray_length=3.0):
    origin = np.asarray(ray.origin, dtype=float)
    direction = np.asarray(ray.direction, dtype=float)
    if not (np.all(np.isfinite(origin)) and np.all(np.isfinite(direction))):
        return

    p1, out = trace_ray(element, ray)

    if p1 is not None:
        _add_segment(view, origin, p1, INCIDENT_COLOR)
        _add_marker(view, p1, INCIDENT_COLOR)
    else:
        _add_segment(view, origin, origin + ray_length * direction, INCIDENT_COLOR)

    if out is None:
        return
    if out.origin is not None and np.all(np.isfinite(out.origin)):
        p2 = np.asarray(out.origin, dtype=float)
        if p1 is not None:
            _add_segment(view, p1, p2, INTERNAL_COLOR)
        _add_marker(view, p2, EXIT_COLOR)
        if np.all(np.isfinite(out.direction)):
            _add_segment(view, p2, p2 + ray_length * np.asarray(out.direction, dtype=float), EXIT_COLOR)


def visualize(element, rays, x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), ray_length=3.0, show=True):
    canvas = scene.SceneCanvas(
        keys="interactive",
        size=(900, 700),
        show=False,
        bgcolor=(1.0, 1.0, 1.0, 1.0),
        title="OpticsSimulator",
    )
    view = canvas.central_widget.add_view(camera="turntable")

    axis_len = max(abs(x_range[0]), abs(x_range[1]), abs(y_range[0]), abs(y_range[1]))
    _add_axes(view, length=axis_len)
    _attach(view, make_surface_visual(element.surface1, SURFACE1_COLOR))
    _attach(view, make_surface_visual(element.surface2, SURFACE2_COLOR))

    sidewall = getattr(element, "sidewall", None)
    if sidewall is not None:
        _attach(view, make_wall_visual(sidewall, SIDEWALL_COLOR))

    if isinstance(rays, (list, tuple)):
        ray_list = list(rays)
    else:
        ray_list = [rays]
    for ray in ray_list:
        draw_ray(view, element, ray, ray_length=ray_length)

    if show:
        canvas.show(run=True)
    return canvas


def main():
    from optics.refraction import Ray, RefractiveElement

    schema_path = Path(__file__).resolve().parent / "tests" / "offset_geometry.json"
    with open(schema_path, "r") as f:
        schema = f.read()

    element = RefractiveElement(schema)
    rays = [
        Ray(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -1.0])),
        Ray(np.array([0.5, 0.0, 3.0]), np.array([0.0, 0.0, -1.0])),
        Ray(np.array([0.0, 0.4, 3.0]), np.array([0.0, 0.0, -1.0])),
    ]
    visualize(element, rays)


if __name__ == "__main__":
    main()
