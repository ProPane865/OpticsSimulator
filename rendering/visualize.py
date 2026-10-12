from collections.abc import Sequence
from typing import Any

import numpy as np
from vispy import scene
from vispy.color import get_colormap
from vispy.scene import visuals

from optics.measurement_plane import MeasurementPlane
from optics.ray import Ray
from optics.refraction import RefractiveElement, trace_stack_path
from optics.sidewall import LensSidewall
from optics.surface import Surface

from .meshing import SurfaceMesher

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


def surface_positions(surface: Surface, n: int = 128) -> "tuple[np.ndarray, np.ndarray]":
    return SurfaceMesher(surface).mesh(n)


def make_surface_visual(surface: Surface, color: Any = SURFACE1_COLOR, **mesh_kwargs: Any) -> Any:
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


def make_wall_visual(sidewall: LensSidewall, color: Any = SIDEWALL_COLOR) -> Any:
    verts, faces = sidewall.mesh()

    mesh = visuals.Mesh(verts, faces, color=color, shading="smooth")
    mesh.set_gl_state(
        blend=True,
        depth_test=True,
        depth_mask=True,
        blend_func=("src_alpha", "one_minus_src_alpha")
    )

    return mesh


def _cell_to_vertex_values(grid: np.ndarray) -> np.ndarray:
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


def _measurement_vertex_colors(plane: MeasurementPlane) -> np.ndarray:
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


def make_measurement_plane_visual(plane: MeasurementPlane) -> Any:
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


def _measurement_outline(plane: MeasurementPlane) -> list[np.ndarray]:
    corners = [
        plane.point(plane.u_min, plane.v_min),
        plane.point(plane.u_max, plane.v_min),
        plane.point(plane.u_max, plane.v_max),
        plane.point(plane.u_min, plane.v_max),
    ]
    return [corners[k] for k in (0, 1, 1, 2, 2, 3, 3, 0)]


def build_measurement_nodes(view: Any, plane: MeasurementPlane) -> list[Any]:
    nodes = [_attach(view, make_measurement_plane_visual(plane))]
    outline = _measurement_outline(plane)
    for a, b in zip(outline[0::2], outline[1::2]):
        segment = _add_segment(view, a, b, MEASUREMENT_OUTLINE_COLOR, width=2)
        if segment is not None:
            nodes.append(segment)
    return nodes


def _marker_positions(
    pos: np.ndarray, radius: float = 0.03, n_phi: int = 16, n_theta: int = 8
) -> "tuple[np.ndarray, np.ndarray]":
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


def _attach(view: Any, visual: Any) -> Any:
    view.add(visual)
    return visual


def _make_marker(pos: np.ndarray, color: Any, radius: float = 0.03) -> Any:
    verts, faces = _marker_positions(pos, radius=radius)
    marker = visuals.Mesh(verts, faces, color=color, shading="flat")
    marker.update_gl_state(depth_test=False)
    return marker


def _add_marker(view: Any, pos: np.ndarray, color: Any, radius: float = 0.03) -> Any:
    return _attach(view, _make_marker(pos, color, radius=radius))


def _make_segment(a: np.ndarray, b: np.ndarray, color: Any, width: int = 3) -> Any:
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    segment = visuals.Line(pos=np.array([a, b]), color=color, width=width)
    segment.update_gl_state(depth_test=False)
    return segment


def _add_segment(view: Any, a: np.ndarray, b: np.ndarray, color: Any, width: int = 3) -> Any:
    segment = _make_segment(a, b, color, width=width)
    if segment is None:
        return None
    return _attach(view, segment)


def _add_axes(view: Any, length: float = 2.0, width: int = 2) -> None:
    origin = np.zeros(3)
    for name, ix in (("x", 0), ("y", 1), ("z", 2)):
        end = np.zeros(3)
        end[ix] = length
        _add_segment(view, origin, end, AXIS_COLORS[name], width=width)


RAY_SEGMENT_COLORS = {
    "incident": INCIDENT_COLOR,
    "internal": INTERNAL_COLOR,
    "exit": EXIT_COLOR,
}


def _collect_ray_segments(
    elements: Sequence[RefractiveElement],
    rays: Sequence[Ray],
    ray_length: float = 3.0,
) -> "tuple[list[tuple[np.ndarray, np.ndarray, str]], list[tuple[np.ndarray, tuple[float, float, float, float]]]]":
    """Return ``(segments, markers)`` for all rays traced through ``elements``.

    ``segments`` is a list of ``(start, end, kind)`` tuples and ``markers`` a
    list of ``(position, color)`` tuples for entry/exit transition points.
    """
    segments = []
    markers = []
    for ray in rays:
        origin = np.asarray(ray.origin, dtype=float)
        direction = np.asarray(ray.direction, dtype=float)
        if not (np.all(np.isfinite(origin)) and np.all(np.isfinite(direction))):
            continue
        ray_segments, _ = trace_stack_path(ray, list(elements), ray_length=ray_length)
        prev_kind = None
        for start, end, kind in ray_segments:
            start = np.asarray(start, dtype=float)
            end = np.asarray(end, dtype=float)
            if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
                prev_kind = kind
                continue
            segments.append((start, end, kind))
            if kind == "internal" and prev_kind != "internal":
                markers.append((start, INCIDENT_COLOR))
            elif prev_kind == "internal" and kind != "internal":
                markers.append((start, EXIT_COLOR))
            prev_kind = kind
    return segments, markers


def _make_markers_mesh(
    markers: "Sequence[tuple[np.ndarray, tuple[float, float, float, float]]]",
    radius: float = 0.03,
) -> Any:
    """Merge many sphere markers into a single mesh with vertex colors."""
    verts_chunks = []
    faces_chunks = []
    colors_chunks = []
    offset = 0
    for pos, color in markers:
        verts, faces = _marker_positions(pos, radius=radius)
        verts_chunks.append(verts)
        faces_chunks.append(faces + offset)
        colors_chunks.append(np.tile(np.asarray(color, dtype=float), (len(verts), 1)))
        offset += len(verts)
    if not verts_chunks:
        return None
    mesh = visuals.Mesh(
        np.vstack(verts_chunks),
        np.vstack(faces_chunks),
        vertex_colors=np.vstack(colors_chunks),
        shading="flat",
    )
    mesh.update_gl_state(depth_test=False)
    return mesh


def build_ray_nodes(
    view: Any,
    elements: "RefractiveElement | Sequence[RefractiveElement]",
    rays: Sequence[Ray],
    ray_length: float = 3.0,
) -> list[Any]:
    """Draw all rays as one batched line node plus one merged marker mesh."""
    if isinstance(elements, RefractiveElement):
        elements = [elements]
    segments, markers = _collect_ray_segments(list(elements), rays, ray_length=ray_length)

    nodes = []
    if segments:
        pos = np.array([[start, end] for start, end, _ in segments], dtype=float)
        colors = np.empty((2 * len(segments), 4), dtype=float)
        for i, (_, _, kind) in enumerate(segments):
            colors[2 * i] = RAY_SEGMENT_COLORS[kind]
            colors[2 * i + 1] = RAY_SEGMENT_COLORS[kind]
        line = visuals.Line(pos=pos, connect="segments", color=colors, width=3)
        line.update_gl_state(depth_test=False)
        nodes.append(_attach(view, line))
    if markers:
        mesh = _make_markers_mesh(markers)
        if mesh is not None:
            nodes.append(_attach(view, mesh))
    return nodes


def create_canvas(width: int = 900, height: int = 700) -> Any:
    return scene.SceneCanvas(
        keys="interactive",
        size=(width, height),
        show=False,
        bgcolor=(1.0, 1.0, 1.0, 1.0),
        title="OpticsSimulator",
    )


def build_element_nodes(view: Any, element: RefractiveElement) -> list[Any]:
    nodes = []
    nodes.append(_attach(view, make_surface_visual(element.surface1, SURFACE1_COLOR)))
    nodes.append(_attach(view, make_surface_visual(element.surface2, SURFACE2_COLOR)))

    sidewall = getattr(element, "sidewall", None)
    if sidewall is not None:
        nodes.append(_attach(view, make_wall_visual(sidewall, SIDEWALL_COLOR)))
    return nodes


def make_source_visual(source: Any, color: Any = SOURCE_COLOR, radius: float = 0.08) -> Any:
    return _make_marker(np.asarray(source.position, dtype=float), color, radius=radius)


def build_source_nodes(view: Any, source: Any, radius: float = 0.08) -> list[Any]:
    return [_attach(view, make_source_visual(source, radius=radius))]


def remove_nodes(nodes: Sequence[Any]) -> None:
    for node in nodes:
        if node is not None:
            node.parent = None


def build_scene(
    view: Any,
    elements: "RefractiveElement | Sequence[RefractiveElement]",
    rays: "Ray | Sequence[Ray]",
    x_range: "tuple[float, float]" = (-2.0, 2.0),
    y_range: "tuple[float, float]" = (-2.0, 2.0),
    ray_length: float = 3.0,
) -> "tuple[list[RefractiveElement], list[Any], list[Any]]":
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
    ray_nodes = build_ray_nodes(view, element_list, ray_list, ray_length=ray_length)
    return element_list, element_nodes, ray_nodes


def clear_scene(view: Any) -> None:
    camera = view.camera
    for node in list(view.scene.children):
        if node is not camera:
            node.parent = None
