"""Type stubs for the compiled native kernel module (``optics._native``)."""

from typing import Any

def compile_surface(
    x: str, y: str, z: str, u_min: float, u_max: float, v_min: float, v_max: float
) -> Any: ...
def build_surface_desc(
    model: Any,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    tu: float,
    tv: float,
    ap_r: float,
    ap_cx: float,
    ap_cy: float,
    rotation: Any,
    translation: Any,
) -> Any: ...
def build_grid(
    p_world: Any,
    uf_valid: Any,
    vf_valid: Any,
    pts: Any,
    finite: Any,
    uf: Any,
    vf: Any,
) -> Any: ...
def build_element_context(
    desc1: Any,
    desc2: Any,
    grid1: Any,
    grid2: Any,
    refractive_index: float,
    lo: Any,
    hi: Any,
    tri: Any,
    normals: Any,
    graph1: bool,
    graph2: bool,
) -> Any: ...
def build_trace_stack(contexts: list[Any]) -> Any: ...

def norm(v: Any) -> float: ...
def normalize(v: Any) -> Any: ...
def rotation_from_z(direction: Any) -> Any: ...

def newton_intersection(
    desc: Any, o: Any, d: Any, u: float, v: float, t: float
) -> tuple[float, float, float, bool]: ...
def surface_intersect(desc: Any, grid: Any, o: Any, d: Any) -> tuple[float, float, float, bool]: ...
def solve_xy(desc: Any, x_target: float, y_target: float, u: float, v: float) -> tuple[float, float, bool]: ...
def parameters_at_xy(desc: Any, grid: Any, x: float, y: float) -> tuple[float, float, bool]: ...
def plane_intersect(ro: Any, d: Any, o: Any, n: Any, eps: float) -> tuple[float, float, float, bool]: ...
def mesh_intersect(
    o: Any, d: Any, tri: Any, det_eps: float, bary_tol: float, min_face_t: float
) -> tuple[float, int, float, float, bool]: ...
def orient(normal: Any, incident: Any) -> Any: ...
def refract_or_reflect(
    incident: Any, normal: Any, n_from: float, n_to: float
) -> tuple[Any, bool]: ...

def trace_stack_segments(
    stack: Any, o: Any, d: Any, ray_length: float, starts: Any, ends: Any, kinds: Any
) -> tuple[int, bool, Any, Any]: ...
def trace_segments_batch(
    stack: Any, origins: Any, dirs: Any, ray_length: float, starts: Any, ends: Any, kinds: Any, counts: Any
) -> None: ...
def trace_stack_to_plane(
    stack: Any,
    o: Any,
    d: Any,
    po: Any,
    pn: Any,
    eps: float,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    out_o: Any,
    out_d: Any,
) -> bool: ...

def accumulate(
    grid: Any,
    u: Any,
    v: Any,
    power: Any,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    ru: int,
    rv: int,
    interval_u: float,
    interval_v: float,
) -> float: ...
def measure_direct(
    origins: Any, dirs: Any, powers: Any, o: Any, n: Any, eps: float, us: Any, vs: Any, pws: Any
) -> None: ...
def measure_stack(
    stack: Any,
    origins: Any,
    dirs: Any,
    powers: Any,
    o: Any,
    n: Any,
    eps: float,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    us: Any,
    vs: Any,
    pws: Any,
) -> None: ...
