import json
from pathlib import Path

import numpy as np
from vispy import scene
from vispy.scene import visuals
from vispy.color import get_colormap

from .meshing import SurfaceMesher
from optics.ray import Ray
from optics.refraction import RefractiveElement, trace_stack_path
from optics.config import stack_from_config

SURFACE1_COLOR = (0.35, 0.65, 1.0, 0.25)
SURFACE2_COLOR = (0.2, 0.45, 0.9, 0.25)
SIDEWALL_COLOR = (0.55, 0.55, 0.65, 0.3)
INCIDENT_COLOR = (0.9, 0.2, 0.2, 1.0)
INTERNAL_COLOR = (1.0, 0.6, 0.1, 1.0)
EXIT_COLOR = (0.2, 0.75, 0.3, 1.0)
SOURCE_COLOR = (1.0, 0.85, 0.25, 1.0)
MEASUREMENT_COLOR = (0.5, 0.5, 0.55, 0.35)
MEASUREMENT_OUTLINE_COLOR = (0.75, 0.35, 0.85, 1.0)
MEASUREMENT_COLORMAP = "viridis"
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
        depth_mask=True,
        blend_func=("src_alpha", "one_minus_src_alpha")
    )
    
    return mesh


def make_wall_visual(sidewall, color=SIDEWALL_COLOR):
    verts, faces = sidewall.mesh()

    mesh = visuals.Mesh(verts, faces, color=color, shading="smooth")
    mesh.set_gl_state(
        blend=True,
        depth_test=True,
        depth_mask=True,
        blend_func=("src_alpha", "one_minus_src_alpha")
    )

    return mesh


def _cell_to_vertex_values(grid):
    """Expand a (rows, cols) cell grid to (rows+1, cols+1) vertex values."""
    rows, cols = grid.shape
    out = np.empty((rows + 1, cols + 1), dtype=float)
    out[1:-1, 1:-1] = 0.25 * (
        grid[:-1, :-1] + grid[:-1, 1:] + grid[1:, :-1] + grid[1:, 1:]
    )
    out[1:-1, 0] = 0.5 * (grid[:-1, 0] + grid[1:, 0])
    out[1:-1, -1] = 0.5 * (grid[:-1, -1] + grid[1:, -1])
    out[0, 1:-1] = 0.5 * (grid[0, :-1] + grid[0, 1:])
    out[-1, 1:-1] = 0.5 * (grid[-1, :-1] + grid[-1, 1:])
    out[0, 0] = grid[0, 0]
    out[0, -1] = grid[0, -1]
    out[-1, 0] = grid[-1, 0]
    out[-1, -1] = grid[-1, -1]
    return out


def _measurement_vertex_colors(plane):
    irradiance = plane.irradiance()
    values = _cell_to_vertex_values(irradiance.T)
    flat = np.asarray(MEASUREMENT_COLOR, dtype=float)
    vmax = float(np.nanmax(values)) if values.size else 0.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        return np.tile(flat, (values.size, 1))
    normalized = np.clip(values / vmax, 0.0, 1.0).ravel()
    colors = np.asarray(
        get_colormap(MEASUREMENT_COLORMAP).map(normalized), dtype=float
    )
    colors[..., 3] = 0.85
    return colors


def make_measurement_plane_visual(plane):
    ru, rv = plane.ru, plane.rv
    us = np.linspace(plane.u_min, plane.u_max, ru + 1)
    vs = np.linspace(plane.v_min, plane.v_max, rv + 1)
    U, V = np.meshgrid(us, vs, indexing="ij")
    verts = plane.point(U, V).reshape(-1, 3)

    i = np.repeat(np.arange(ru), rv)
    j = np.tile(np.arange(rv), ru)
    a = i * (rv + 1) + j
    b = (i + 1) * (rv + 1) + j
    c = (i + 1) * (rv + 1) + (j + 1)
    d = i * (rv + 1) + (j + 1)
    faces = np.vstack([
        np.column_stack([a, b, d]),
        np.column_stack([b, c, d]),
    ])

    colors = _measurement_vertex_colors(plane)
    mesh = visuals.Mesh(verts, faces, vertex_colors=colors, shading="flat")
    mesh.set_gl_state(
        blend=True,
        depth_test=True,
        depth_mask=True,
        blend_func=("src_alpha", "one_minus_src_alpha"),
    )
    return mesh


def _measurement_outline(plane):
    corners = [
        plane.point(plane.u_min, plane.v_min),
        plane.point(plane.u_max, plane.v_min),
        plane.point(plane.u_max, plane.v_max),
        plane.point(plane.u_min, plane.v_max),
    ]
    return [corners[k] for k in (0, 1, 1, 2, 2, 3, 3, 0)]


def build_measurement_nodes(view, plane):
    nodes = [_attach(view, make_measurement_plane_visual(plane))]
    outline = _measurement_outline(plane)
    for a, b in zip(outline[0::2], outline[1::2]):
        segment = _add_segment(view, a, b, MEASUREMENT_OUTLINE_COLOR, width=2)
        if segment is not None:
            nodes.append(segment)
    return nodes


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


def draw_ray_stack(view, elements, ray, ray_length=3.0):
    nodes = []
    origin = np.asarray(ray.origin, dtype=float)
    direction = np.asarray(ray.direction, dtype=float)
    if not (np.all(np.isfinite(origin)) and np.all(np.isfinite(direction))):
        return nodes

    segments, _ = trace_stack_path(ray, list(elements), ray_length=ray_length)
    colors = {
        "incident": INCIDENT_COLOR,
        "internal": INTERNAL_COLOR,
        "exit": EXIT_COLOR,
    }

    prev_kind = None
    for start, end, kind in segments:
        segment = _add_segment(view, start, end, colors[kind])
        if segment is not None:
            nodes.append(segment)
        if kind == "internal" and prev_kind != "internal":
            nodes.append(_add_marker(view, start, INCIDENT_COLOR))
        elif prev_kind == "internal" and kind != "internal":
            nodes.append(_add_marker(view, start, EXIT_COLOR))
        prev_kind = kind
    return nodes


def draw_ray(view, elements, ray, ray_length=3.0):
    if isinstance(elements, RefractiveElement):
        elements = [elements]
    return draw_ray_stack(view, list(elements), ray, ray_length=ray_length)


def create_canvas(width=900, height=700):
    return scene.SceneCanvas(
        keys="interactive",
        size=(width, height),
        show=False,
        bgcolor=(1.0, 1.0, 1.0, 1.0),
        title="OpticsSimulator",
    )


def build_element_nodes(view, element):
    nodes = []
    nodes.append(_attach(view, make_surface_visual(element.surface1, SURFACE1_COLOR)))
    nodes.append(_attach(view, make_surface_visual(element.surface2, SURFACE2_COLOR)))

    sidewall = getattr(element, "sidewall", None)
    if sidewall is not None:
        nodes.append(_attach(view, make_wall_visual(sidewall, SIDEWALL_COLOR)))
    return nodes


def make_source_visual(source, color=SOURCE_COLOR, radius=0.08):
    return _make_marker(np.asarray(source.position, dtype=float), color, radius=radius)


def build_source_nodes(view, source, radius=0.08):
    return [_attach(view, make_source_visual(source, radius=radius))]


def remove_nodes(nodes):
    for node in nodes:
        if node is not None:
            node.parent = None


def build_scene(view, elements, rays, x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), ray_length=3.0):
    if isinstance(elements, RefractiveElement):
        element_list = [elements]
    else:
        element_list = list(elements)

    axis_len = max(abs(x_range[0]), abs(x_range[1]), abs(y_range[0]), abs(y_range[1]))
    _add_axes(view, length=axis_len)
    element_nodes = []
    for element in element_list:
        element_nodes.append(build_element_nodes(view, element))

    if isinstance(rays, (list, tuple)):
        ray_list = list(rays)
    else:
        ray_list = [rays]
    ray_nodes = []
    for ray in ray_list:
        nodes = draw_ray(view, element_list, ray, ray_length=ray_length)
        if nodes:
            ray_nodes.extend(nodes)
    return element_list, element_nodes, ray_nodes


def clear_scene(view):
    camera = view.camera
    for node in list(view.scene.children):
        if node is not camera:
            node.parent = None


def visualize(elements, rays, x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), ray_length=3.0, show=True):
    canvas = create_canvas()
    view = canvas.central_widget.add_view(camera="turntable")
    build_scene(view, elements, rays, x_range=x_range, y_range=y_range, ray_length=ray_length)

    if show:
        canvas.show(run=True)
    return canvas


def default_stack():
    schema_path = Path(__file__).resolve().parent.parent / "tests" / "test_stack.json"
    with open(schema_path, "r") as f:
        stack = json.load(f)
    return stack_from_config(stack)


def default_rays():
    return [
        Ray(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0),
        Ray(np.array([0.5, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0),
        Ray(np.array([0.0, 0.4, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0),
    ]


def main():
    visualize(default_stack(), default_rays(), ray_length=6.0)


if __name__ == "__main__":
    main()
