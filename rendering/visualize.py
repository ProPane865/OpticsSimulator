from collections.abc import Sequence
from typing import Any

import numpy as np
from vispy import scene
from vispy.color import get_colormap
from vispy.geometry import MeshData
from vispy.scene import visuals

from optics.element import RefractiveElement
from optics.measurement_plane import MeasurementPlane, MeasurementResult
from optics.primitive import LensSidewall, Ray, Surface
from optics.system import OpticalSystem

from .meshing import SurfaceMesher

SURFACE1_COLOR = (0.35, 0.65, 1.0, 0.5)
SURFACE2_COLOR = (0.2, 0.45, 0.9, 0.5)
SIDEWALL_COLOR = (0.55, 0.55, 0.65, 0.6)

SOURCE_COLOR = (1.0, 0.85, 0.25, 1.0)

INCIDENT_COLOR = (1.0, 1.0, 0.0, 0.2)
INTERNAL_COLOR = (1.0, 1.0, 0.0, 0.2)
EXIT_COLOR = (1.0, 1.0, 0.0, 0.2)

MEASUREMENT_COLOR = (0.5, 0.5, 0.55, 0.35)
MEASUREMENT_OUTLINE_COLOR = (0.75, 0.35, 0.85, 1.0)
MEASUREMENT_COLORMAP = "viridis"
MEASUREMENT_MESH_ORDER = -10

SHOW_RAY_INTERSECTION_MARKERS = False
USE_DEBUG_COLORS = False

AXIS_COLORS = {
    "x": (0.85, 0.15, 0.15, 1.0),
    "y": (0.15, 0.65, 0.2, 1.0),
    "z": (0.15, 0.3, 0.85, 1.0),
}

if USE_DEBUG_COLORS:
    INCIDENT_COLOR = (0.9, 0.2, 0.2, 1.0)
    INTERNAL_COLOR = (1.0, 0.6, 0.1, 1.0)
    EXIT_COLOR = (0.2, 0.75, 0.3, 1.0)


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


_COLORMAP_LUT_SIZE = 256
_COLORMAP_LUTS: "dict[str, np.ndarray]" = {}


def _colormap_lut(name: str) -> np.ndarray:
    """Cached 256-entry RGBA LUT for ``name``, alpha pre-applied.

    Sampling a LUT is orders of magnitude faster than vispy's per-value
    ``Colormap.map`` on full detector grids, and visually identical at
    256 quantization levels.
    """
    lut = _COLORMAP_LUTS.get(name)
    if lut is None:
        lut = np.asarray(
            get_colormap(name).map(np.linspace(0.0, 1.0, _COLORMAP_LUT_SIZE)),
            dtype=np.float32,
        )
        lut[..., 3] = 0.85
        _COLORMAP_LUTS[name] = lut
    return lut


def _measurement_vertex_colors(result: MeasurementResult) -> np.ndarray:
    irradiance = result.irradiance()
    values = _cell_to_vertex_values(irradiance.T)
    vmax = float(np.nanmax(values)) if values.size else 0.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        return np.tile(
            np.asarray(MEASUREMENT_COLOR, dtype=np.float32), (values.size, 1)
        )
    normalized = np.clip(values / vmax, 0.0, 1.0).ravel()
    lut = _colormap_lut(MEASUREMENT_COLORMAP)
    return lut[np.rint(normalized * (len(lut) - 1)).astype(np.intp)]


def _measurement_plane_geometry(plane: MeasurementPlane) -> "tuple[np.ndarray, np.ndarray]":
    """Vertex grid and quad faces for a plane's detector surface."""
    ru, rv = plane.ru, plane.rv
    us = np.linspace(plane.u_min, plane.u_max, ru + 1)
    vs = np.linspace(plane.v_min, plane.v_max, rv + 1)
    U, V = np.meshgrid(us, vs, indexing="ij")
    verts = plane.point(U, V).reshape(-1, 3).astype(np.float32, copy=False)

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
    return verts, faces


def make_measurement_plane_visual(plane: MeasurementPlane, result: MeasurementResult) -> Any:
    verts, faces = _measurement_plane_geometry(plane)
    colors = _measurement_vertex_colors(result)
    mesh = visuals.Mesh(verts, faces, vertex_colors=colors, shading="flat")
    mesh.set_gl_state(
        blend=True,
        depth_test=True,
        depth_mask=True,
        blend_func=("src_alpha", "one_minus_src_alpha"),
    )
    return mesh


def measurement_geometry_key(plane: MeasurementPlane) -> tuple:
    """Plane geometry that the mesh and outline depend on (result-free)."""
    return (
        (float(plane.o[0]), float(plane.o[1]), float(plane.o[2])),
        (float(plane.n[0]), float(plane.n[1]), float(plane.n[2])),
        plane.u_min,
        plane.u_max,
        plane.v_min,
        plane.v_max,
        int(plane.ru),
        int(plane.rv),
    )


MeshTemplate = tuple[np.ndarray, np.ndarray, "tuple[np.ndarray, ...] | None"]


def measurement_mesh_template(mesh: Any) -> "MeshTemplate":
    """Geometry and precomputed normals behind a measurement mesh.

    The arrays back color-only refreshes (:func:`update_measurement_colors`):
    the detector grid geometry never changes between refreshes, so computing
    vispy's vertex/face normals once here (instead of on every data change and
    draw) keeps per-trace refreshes cheap.
    """
    meshdata = mesh.mesh_data
    verts = meshdata.get_vertices()
    faces = meshdata.get_faces()
    try:
        normals: tuple[np.ndarray, ...] | None = (
            meshdata.get_vertex_normals(),
            meshdata.get_vertex_normals(indexed="faces"),
            meshdata.get_face_normals(),
            meshdata.get_face_normals(indexed="faces"),
        )
    except Exception:
        normals = None
    return verts, faces, normals


def update_measurement_colors(mesh: Any, result: MeasurementResult, template: "MeshTemplate") -> None:
    """Re-tint an existing measurement mesh in place with fresh detector data.

    Replaces only the vertex colors on a geometry-identical ``MeshData``; the
    template's cached normals are seeded onto the new ``MeshData`` because
    vispy re-derives them on every data change otherwise, which costs more
    than the rest of a refresh combined.
    """
    verts, faces, normals = template
    meshdata = MeshData(
        vertices=verts,
        faces=faces,
        vertex_colors=_measurement_vertex_colors(result),
    )
    if normals is not None and "_vertex_normals" in meshdata.__dict__:
        # vispy keeps normal caches per MeshData; the geometry is identical,
        # so carry them across (private fields; skipped if internals change).
        (
            meshdata._vertex_normals,
            meshdata._vertex_normals_indexed_by_faces,
            meshdata._face_normals,
            meshdata._face_normals_indexed_by_faces,
        ) = normals
    mesh.set_data(meshdata=meshdata)


def _measurement_outline(plane: MeasurementPlane) -> list[np.ndarray]:
    corners = [
        plane.point(plane.u_min, plane.v_min),
        plane.point(plane.u_max, plane.v_min),
        plane.point(plane.u_max, plane.v_max),
        plane.point(plane.u_min, plane.v_max),
    ]
    return [corners[k] for k in (0, 1, 1, 2, 2, 3, 3, 0)]


def build_measurement_nodes(
    view: Any, plane: MeasurementPlane, result: MeasurementResult | None = None
) -> list[Any]:
    plane_mesh = make_measurement_plane_visual(plane, result if result is not None else plane.new_result())
    plane_mesh.order = MEASUREMENT_MESH_ORDER
    nodes = [_attach(view, plane_mesh)]
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


def _make_segment(a: np.ndarray, b: np.ndarray, color: Any, width: int) -> Any:
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    segment = visuals.Line(pos=np.array([a, b]), color=color, width=width)
    segment.update_gl_state(depth_test=False)
    return segment


def _add_segment(view: Any, a: np.ndarray, b: np.ndarray, color: Any, width: int) -> Any:
    segment = _make_segment(a, b, color, width=width)
    if segment is None:
        return None
    return _attach(view, segment)


def _add_axes(view: Any, width: int, length: float) -> None:
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


_SEGMENT_KIND_NAMES = ("incident", "internal", "exit")


def _append_ray_path(
    segments: list,
    markers: list,
    ray_segments: "Sequence[tuple[np.ndarray, np.ndarray, str]]",
) -> None:
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


def _as_system(
    target: "OpticalSystem | RefractiveElement | Sequence[RefractiveElement]",
) -> OpticalSystem:
    """Normalize a ray-tracing target to an :class:`OpticalSystem`.

    Rendering entry points accept a persistent system (the app's path) or a
    plain element stack (demos and tests), which is wrapped on the fly; the
    native element state behind it is cached per element either way.
    """
    if isinstance(target, OpticalSystem):
        return target
    if isinstance(target, RefractiveElement):
        return OpticalSystem([target])
    return OpticalSystem(target)


def _ray_line_data(
    starts: np.ndarray,
    ends: np.ndarray,
    kinds: np.ndarray,
    counts: np.ndarray,
) -> "tuple[np.ndarray | None, np.ndarray | None]":
    """Vectorized ``(pos, colors)`` for the batched ray Line visual.

    Takes the trace buffers ``(N, C, 3)``/``(N, C)``/``(N,)`` and drops every
    slot beyond a ray's segment count or holding a non-finite endpoint;
    returns ``(None, None)`` when nothing is drawable.
    """
    cap = starts.shape[1]
    valid = np.arange(cap)[None, :] < np.asarray(counts).reshape(-1, 1)
    valid &= np.isfinite(starts).all(axis=-1)
    valid &= np.isfinite(ends).all(axis=-1)
    if not valid.any():
        return None, None
    pos = np.empty((int(valid.sum()), 2, 3), dtype=float)
    pos[:, 0] = starts[valid]
    pos[:, 1] = ends[valid]
    kind_colors = np.asarray(
        [RAY_SEGMENT_COLORS[name] for name in _SEGMENT_KIND_NAMES], dtype=float
    )
    colors = np.repeat(kind_colors[np.asarray(kinds)[valid]], 2, axis=0)
    return pos, colors


def _ray_markers(
    starts: np.ndarray,
    ends: np.ndarray,
    kinds: np.ndarray,
    counts: np.ndarray,
) -> "list[tuple[np.ndarray, tuple[float, float, float, float]]]":
    """Entry/exit transition markers per ray (debug view; per-ray Python loop)."""
    segments: list = []
    markers: list = []
    for i in range(counts.shape[0]):
        _append_ray_path(
            segments,
            markers,
            [
                (
                    starts[i, k].copy(),
                    ends[i, k].copy(),
                    _SEGMENT_KIND_NAMES[int(kinds[i, k])],
                )
                for k in range(int(counts[i]))
            ],
        )
    return markers


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
    mesh.update_gl_state(depth_test=True)
    return mesh


def send_to_back(nodes: Sequence[Any], base: int = -2) -> None:
    """Draw the given nodes before their siblings so surfaces blend over them."""
    for offset, node in enumerate(nodes):
        if node is not None:
            node.order = base + offset


def _build_ray_visual_nodes(
    view: Any,
    pos: "np.ndarray | None",
    colors: "np.ndarray | None",
    markers: "Sequence[tuple[np.ndarray, tuple[float, float, float, float]]]",
    show_markers: bool,
) -> list[Any]:
    nodes = []
    if pos is not None:
        line = visuals.Line(pos=pos, connect="segments", color=colors, width=1)
        line.update_gl_state(depth_test=True, depth_mask=True)
        nodes.append(_attach(view, line))
    if markers and show_markers:
        mesh = _make_markers_mesh(markers)
        if mesh is not None:
            nodes.append(_attach(view, mesh))
    send_to_back(nodes)
    return nodes


def build_ray_nodes(
    view: Any,
    system: "OpticalSystem | RefractiveElement | Sequence[RefractiveElement]",
    rays: Sequence[Ray],
    ray_length: float = 3.0,
    show_markers: "bool | None" = None,
) -> list[Any]:
    """Draw all rays as one batched line node plus one merged marker mesh.

    Traces through the given :class:`~optics.system.OpticalSystem` (or wraps
    a plain element stack). Entry/exit intersection markers are disabled by
    default; set ``SHOW_RAY_INTERSECTION_MARKERS`` to ``True`` or pass
    ``show_markers`` to enable them.
    """
    if show_markers is None:
        show_markers = SHOW_RAY_INTERSECTION_MARKERS
    if not rays:
        return []
    origins = np.asarray([r.origin for r in rays], dtype=float)
    dirs = np.asarray([r.direction for r in rays], dtype=float)
    finite = np.isfinite(origins).all(axis=1) & np.isfinite(dirs).all(axis=1)
    if not finite.any():
        return []
    starts, ends, kinds, counts = _as_system(system).trace_segments(
        origins[finite], dirs[finite], ray_length=ray_length
    )
    pos, colors = _ray_line_data(starts, ends, kinds, counts)
    markers = _ray_markers(starts, ends, kinds, counts) if show_markers else []
    return _build_ray_visual_nodes(view, pos, colors, markers, show_markers)


def build_ray_nodes_from_segments(
    view: Any,
    starts: np.ndarray,
    ends: np.ndarray,
    kinds: np.ndarray,
    counts: np.ndarray,
    show_markers: "bool | None" = None,
) -> list[Any]:
    """Draw rays from a precomputed batch trace (``starts``/``ends``/``kinds``/
    ``counts``), avoiding a second trace when measurement and rendering share
    one walk."""
    if show_markers is None:
        show_markers = SHOW_RAY_INTERSECTION_MARKERS
    pos, colors = _ray_line_data(starts, ends, kinds, counts)
    markers = _ray_markers(starts, ends, kinds, counts) if show_markers else []
    return _build_ray_visual_nodes(view, pos, colors, markers, show_markers)


def create_canvas(width: int = 900, height: int = 700) -> Any:
    return scene.SceneCanvas(
        keys="interactive",
        size=(width, height),
        show=False,
        bgcolor=(0.1, 0.1, 0.1, 1.0),
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
    _add_axes(view, width=2, length=axis_len)
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
