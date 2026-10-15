import math
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
from numba import njit

from . import constants as _c
from .numerical._kernels import (
    D_ROT,
    D_TRANS,
    E_DESC1,
    E_DESC2,
    E_GRAPH1,
    E_GRAPH2,
    E_GRID1,
    E_GRID2,
    E_HI,
    E_INDEX,
    E_LO,
    E_NORMALS,
    E_TRI,
    build_descriptor,
    build_element_context,
    build_grid,
)
from .primitive.hit import RayHit
from .primitive.material import Material
from .numerical.plane import _intersect_kernel
from .primitive.ray import Ray
from .primitive.sidewall import LensSidewall, _mt_kernel
from .numerical.solver import (
    _HIT_TOL_CONV,
    _HIT_TOL_EXH,
    _HIT_TOL_STALL,
    _accepts_njit,
    _derivatives_njit,
    _eval_njit,
    _norm,
    _parameters_at_xy_njit,
    _surface_intersect_njit,
)
from .primitive.surface import Surface
from .transform import Transform
from .numerical.vector import _normalize

_SIDEWALL_UNSET = object()

_EPSILON = _c.RAY_EPSILON
_MIN_HIT_T = _c.MIN_HIT_T
_MAX_INTERNAL_BOUNCES = _c.MAX_INTERNAL_BOUNCES


@njit(cache=True)
def _reflect_direction(incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return _normalize(incident - 2.0 * float(np.dot(normal, incident)) * normal)


@njit(cache=True)
def _refract_direction(
    incident: np.ndarray, normal: np.ndarray, eta: float, cos_i: float, k: float
) -> np.ndarray:
    return _normalize(eta * incident + (np.sqrt(k) - eta * cos_i) * normal)


@njit(cache=True)
def _orient(normal: np.ndarray, incident: np.ndarray) -> np.ndarray:
    """Align the surface normal with the incident ray direction."""
    if float(np.dot(normal, incident)) < 0.0:
        return -normal
    return normal


@njit(cache=True)
def refract_or_reflect(
    incident: np.ndarray, normal: np.ndarray, n_from: float, n_to: float
) -> "tuple[np.ndarray, bool]":
    """Return ``(direction, refracted)`` for one interface interaction.

    ``normal`` must be oriented so that ``dot(normal, incident) >= 0``.
    On total internal reflection the reflected direction is returned with
    ``refracted=False``.
    """
    eta = n_from / n_to
    cos_i = float(np.dot(normal, incident))
    k = 1.0 - eta**2 * (1.0 - cos_i**2)

    if k < 0.0:
        return _reflect_direction(incident, normal), False
    return _refract_direction(incident, normal, eta, cos_i, k), True


@njit(cache=True)
def trace_direction(incident: np.ndarray, normal: np.ndarray, n_from: float, n_to: float) -> np.ndarray:
    return refract_or_reflect(incident, normal, n_from, n_to)[0]


# --- Nopython element tracing ------------------------------------------------
# Element context tuple layout (and its index constants) live in ``_kernels.py``.

_MAX_PATH_PTS = _MAX_INTERNAL_BOUNCES + 2
_MAX_SEGS_PER_ELEMENT = 2 * _MAX_INTERNAL_BOUNCES + 3
_PLANE_EPS = _c.PLANE_EPS
_DET_SINGULAR_EPS = _c.DET_SINGULAR_EPS
_BARY_TOL = _c.BARY_TOL
_MIN_FACE_T = _c.MIN_FACE_T


@njit(cache=True)
def _normal_njit(desc: Any, u: float, v: float) -> "tuple[float, float, float, bool]":
    """World-space unit normal at (u, v); mirrors ``Surface.normal``."""
    if not _accepts_njit(desc, u, v):
        return 0.0, 0.0, 0.0, False
    ru, rv = _derivatives_njit(desc, u, v)
    n = -np.cross(ru, rv)
    norm = _norm(n)
    if not math.isfinite(norm) or norm <= 0.0:
        return 0.0, 0.0, 0.0, False
    w = desc[D_ROT] @ (n / norm)
    return float(w[0]), float(w[1]), float(w[2]), True


@njit(cache=True)
def _surface_hit_njit(
    desc: Any, grid: Any, o: np.ndarray, d: np.ndarray, min_t: float
) -> "tuple[bool, float, np.ndarray, np.ndarray]":
    """Nearest surface hit with normal, or ``(False, inf, 0, 0)``."""
    t, u, v, ok = _surface_intersect_njit(
        desc, grid, o, d, _HIT_TOL_CONV, _HIT_TOL_STALL, _HIT_TOL_EXH
    )
    if ok and t > min_t:
        nx, ny, nz, nok = _normal_njit(desc, u, v)
        if nok:
            bn = np.zeros(3)
            bn[0] = nx
            bn[1] = ny
            bn[2] = nz
            return True, t, o + t * d, bn
    return False, np.inf, np.zeros(3), np.zeros(3)


@njit(cache=True)
def _element_first_hit(
    elem: Any, o: np.ndarray, d: np.ndarray, min_t: float
) -> "tuple[bool, float, np.ndarray, np.ndarray]":
    """Nearest face hit (surfaces + sidewall); mirrors ``_candidate_hits``."""
    best_t = np.inf
    found = False
    bp = np.zeros(3)
    bn = np.zeros(3)
    f0, t0, p0, n0 = _surface_hit_njit(elem[E_DESC1], elem[E_GRID1], o, d, min_t)
    if f0 and t0 < best_t:
        best_t = t0
        found = True
        bp = p0
        bn = n0
    f1, t1, p1, n1 = _surface_hit_njit(elem[E_DESC2], elem[E_GRID2], o, d, min_t)
    if f1 and t1 < best_t:
        best_t = t1
        found = True
        bp = p1
        bn = n1
    tri = elem[E_TRI]
    if tri.shape[0] > 0:
        t, face, bu, bv, ok = _mt_kernel(
            o, d, tri, _DET_SINGULAR_EPS, _BARY_TOL, _MIN_FACE_T
        )
        if ok and t > min_t and t < best_t:
            best_t = t
            found = True
            bp = o + t * d
            bn = elem[E_NORMALS][int(face)]
    return found, best_t, bp, bn


@njit(cache=True)
def _local_surface_z_njit(
    desc: Any, grid: Any, graph: bool, local: np.ndarray
) -> float:
    """Local height of a surface above ``(local[0], local[1])``."""
    u = float(local[0])
    v = float(local[1])
    if graph:
        if not _accepts_njit(desc, u, v):
            return math.nan
    else:
        uu, vv, ok = _parameters_at_xy_njit(desc, grid, u, v)
        if not ok:
            return math.nan
        u = uu
        v = vv
    z = float(_eval_njit(desc, u, v)[2])
    return z if math.isfinite(z) else math.nan


@njit(cache=True)
def _contains_point_njit(elem: Any, p: np.ndarray) -> bool:
    """Nopython ``RefractiveElement.contains_point``."""
    desc = elem[E_DESC1]
    R = desc[D_ROT]
    T = desc[D_TRANS]
    local = R.T @ (p - T)
    if not np.all(np.isfinite(local)):
        return False
    lo = elem[E_LO]
    hi = elem[E_HI]
    for i in range(3):
        if local[i] < lo[i] or local[i] > hi[i]:
            return False
    z1 = _local_surface_z_njit(elem[E_DESC1], elem[E_GRID1], elem[E_GRAPH1], local)
    z2 = _local_surface_z_njit(elem[E_DESC2], elem[E_GRID2], elem[E_GRAPH2], local)
    if math.isnan(z1) or math.isnan(z2):
        return False
    return bool(min(z1, z2) < local[2] < max(z1, z2))


@njit(cache=True)
def _trace_element_njit(
    elem: Any, o: np.ndarray, d: np.ndarray, pts: np.ndarray
) -> "tuple[np.ndarray, np.ndarray, bool, int, bool, np.ndarray]":
    """Nopython ``RefractiveElement.trace`` for one closed element.

    Writes the internal polyline (entry point, bounce points, exit point)
    into ``pts`` and returns ``(exit_o, exit_d, exited, n_pts, has_incident,
    incident_pt)``.
    """
    inside = _contains_point_njit(elem, o)
    found, t0, hp, hn = _element_first_hit(elem, o, d, 0.0)
    if not found:
        if inside:
            pts[0] = o
            return o, d, True, 1, False, o
        return o, d, False, 0, False, o
    n = elem[E_INDEX]
    if inside:
        cur_o = o
        cur_d = d
        has_inc = False
        ipt = o
        pts[0] = o
        n_pts = 1
    else:
        has_inc = True
        ipt = hp
        d2, _ = refract_or_reflect(d, _orient(hn, d), 1.0, n)
        cur_o = hp + _EPSILON * d2
        cur_d = d2
        pts[0] = hp
        n_pts = 1
    for _ in range(_MAX_INTERNAL_BOUNCES):
        ffound, t1, fp, fn = _element_first_hit(elem, cur_o, cur_d, _MIN_HIT_T)
        if not ffound:
            return cur_o, cur_d, False, n_pts, has_inc, ipt
        pts[n_pts] = fp
        n_pts += 1
        d2, refracted = refract_or_reflect(cur_d, _orient(fn, cur_d), n, 1.0)
        if refracted:
            return fp, d2, True, n_pts, has_inc, ipt
        cur_o = fp + _EPSILON * d2
        cur_d = d2
    return cur_o, cur_d, False, n_pts, has_inc, ipt


@njit(cache=True)
def _trace_stack_segments_njit(
    stack: Any,
    o: np.ndarray,
    d: np.ndarray,
    ray_length: float,
    starts: np.ndarray,
    ends: np.ndarray,
    kinds: np.ndarray,
) -> "tuple[int, bool, np.ndarray, np.ndarray]":
    """Nopython ``trace_stack_path``; segment kinds 0/1/2 = incident/internal/exit."""
    m = len(stack)
    visited = np.zeros(m, np.bool_)
    cur_o = o
    cur_d = d
    kind = 0
    count = 0
    pts = np.empty((_MAX_PATH_PTS, 3))
    cap = starts.shape[0]
    while True:
        best_i = -1
        best_t = np.inf
        i = 0
        for elem in stack:
            if not visited[i]:
                found, t, hp, hn = _element_first_hit(elem, cur_o, cur_d, 0.0)
                if found and t < best_t:
                    best_t = t
                    best_i = i
            i += 1
        if best_i < 0:
            break
        visited[best_i] = True
        exit_o = cur_o
        exit_d = cur_d
        exited = False
        n_pts = 0
        has_inc = False
        ipt = cur_o
        i = 0
        for elem in stack:
            if i == best_i:
                exit_o, exit_d, exited, n_pts, has_inc, ipt = _trace_element_njit(
                    elem, cur_o, cur_d, pts
                )
            i += 1
        if has_inc and count < cap:
            starts[count] = cur_o
            ends[count] = ipt
            kinds[count] = kind
            count += 1
        for k in range(n_pts - 1):
            if count >= cap:
                break
            starts[count] = pts[k]
            ends[count] = pts[k + 1]
            kinds[count] = 1
            count += 1
        if not exited:
            return count, False, cur_o, cur_d
        cur_o = exit_o
        cur_d = exit_d
        kind = 2
    if count < cap:
        starts[count] = cur_o
        ends[count] = cur_o + ray_length * cur_d
        kinds[count] = kind
        count += 1
    return count, True, cur_o, cur_d


@njit(cache=True)
def _trace_stack_to_plane_njit(
    stack: Any,
    o: np.ndarray,
    d: np.ndarray,
    po: np.ndarray,
    pn: np.ndarray,
    eps: float,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    out_o: np.ndarray,
    out_d: np.ndarray,
) -> bool:
    """Nopython ``MeasurementPlane._trace_to_plane``; writes the reaching ray."""
    m = len(stack)
    visited = np.zeros(m, np.bool_)
    cur_o = o
    cur_d = d
    pts = np.empty((_MAX_PATH_PTS, 3))
    while True:
        _, _, t_max, ok = _intersect_kernel(cur_o, cur_d, po, pn, eps)
        if not ok:
            break
        best_i = -1
        best_t = np.inf
        i = 0
        for elem in stack:
            if not visited[i]:
                found, t, hp, hn = _element_first_hit(elem, cur_o, cur_d, 0.0)
                if found and t < t_max and t < best_t:
                    best_t = t
                    best_i = i
            i += 1
        if best_i < 0:
            break
        visited[best_i] = True
        exit_o = cur_o
        exit_d = cur_d
        exited = False
        n_pts = 0
        i = 0
        for elem in stack:
            if i == best_i:
                exit_o, exit_d, exited, n_pts, _, _ = _trace_element_njit(
                    elem, cur_o, cur_d, pts
                )
            i += 1
        for k in range(n_pts - 1):
            seg_o = pts[k]
            diff = pts[k + 1] - seg_o
            seg_len = _norm(diff)
            if seg_len <= 0.0:
                continue
            seg_d = diff / seg_len
            u, v, t, pok = _intersect_kernel(seg_o, seg_d, po, pn, eps)
            if (
                pok
                and u_min <= u <= u_max
                and v_min <= v <= v_max
                and t <= seg_len
            ):
                out_o[:] = seg_o
                out_d[:] = seg_d
                return True
        if not exited:
            return False
        cur_o = exit_o
        cur_d = exit_d
    out_o[:] = cur_o
    out_d[:] = cur_d
    return True


_NULL_ELEMENT_CTX: "tuple[Any, ...] | None" = None

# The stack argument of the trace kernels is a Numba tuple whose *length* is
# part of the static signature: without padding, adding one element changes
# the tuple type and forces a full re-JIT of the whole stack kernel chain
# (~10-20 s) on the calling thread. Contexts are therefore padded to a fixed
# capacity with inert elements so the signature depends only on the capacity.
# The kernels iterate the tuple statically, so compile cost grows with the
# capacity; 4 keeps the one-time compile on par with tracing a 4-element
# stack directly, and capacities only change every 4th stack size.
_CTX_PAD = 4


def _padded_capacity(n_elements: int) -> int:
    return max(_CTX_PAD, -(-n_elements // _CTX_PAD) * _CTX_PAD)


def _null_element_context() -> "tuple[Any, ...]":
    """Inert element context used to pad trace stacks to a fixed capacity.

    Its seed grids are empty and its local bounds are inverted, so the
    kernels' hit/containment scans exit immediately and the descriptor's
    cfuncs are never called; they only have to carry the right types.
    """
    global _NULL_ELEMENT_CTX
    if _NULL_ELEMENT_CTX is None:
        from .primitive.surface import null_surface_cfuncs

        eval_cfn, deriv_cfn = null_surface_cfuncs()
        desc = build_descriptor(
            eval_cfn,
            deriv_cfn,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            -1.0,
            0.0,
            0.0,
            np.eye(3, 3),
            np.zeros(3),
        )
        grid = build_grid(
            np.zeros((0, 3)),
            np.zeros(0),
            np.zeros(0),
            np.zeros((0, 3)),
            np.zeros(0, dtype=bool),
            np.zeros(0),
            np.zeros(0),
        )
        _NULL_ELEMENT_CTX = build_element_context(
            desc,
            desc,
            grid,
            grid,
            1.0,
            np.ones(3),
            -np.ones(3),
            np.zeros((0, 3, 3)),
            np.zeros((0, 3)),
            True,
            True,
        )
    return _NULL_ELEMENT_CTX


def build_trace_context(
    elements: Sequence["RefractiveElement"],
) -> "tuple[Any, ...] | None":
    """Flatten an element stack into plain data for the nopython kernels.

    The returned tuple is padded with inert contexts up to
    ``_padded_capacity(len(elements))`` so the stack-tuple length stays a
    compile-time constant across element additions (see ``_CTX_PAD``).
    Callers that need the real stack size (e.g. segment buffers) must use
    ``len(elements)``, not ``len`` of the returned tuple.

    Returns ``None`` when the stack is empty or any surface lacks the
    nopython codegen path (or uses an unsupported aperture), so callers can
    fall back to the Python tracer.
    """
    if not elements:
        return None
    ctxs: list[Any] = []
    for el in elements:
        d1 = el.surface1.njit_descriptor
        d2 = el.surface2.njit_descriptor
        if d1 is None or d2 is None:
            return None
        g1 = el.surface1.solver.njit_grid()
        g2 = el.surface2.solver.njit_grid()
        wall = el.sidewall
        if wall is None:
            tri = np.empty((0, 3, 3), dtype=float)
            normals = np.empty((0, 3), dtype=float)
        else:
            tri = wall.triangles()
            normals = wall.face_normals()
        lo, hi = el._local_bounds()
        ctxs.append(
            build_element_context(
                d1,
                d2,
                g1,
                g2,
                float(el.material.refractive_index),
                lo,
                hi,
                tri,
                normals,
                el.surface1.is_graph_over_xy,
                el.surface2.is_graph_over_xy,
            )
        )
    null = _null_element_context()
    target = _padded_capacity(len(ctxs))
    ctxs.extend([null] * (target - len(ctxs)))
    return tuple(ctxs)


def _segments_capacity(n_elements: int) -> int:
    return _MAX_SEGS_PER_ELEMENT * n_elements + 2


def _run_in_chunks(work: Callable[[int, int], None], n: int) -> None:
    """Split ``range(n)`` across GIL-free njit chunk calls on threads.

    ``prange`` cannot carry the element-stack tuple through its gufunc
    packing, so stack kernels are dispatched as disjoint slices instead;
    ``nogil=True`` lets the compiled chunks run truly in parallel.
    """
    if n <= 0:
        return
    nt = max(1, min(numba.get_num_threads(), n))
    chunk = -(-n // nt)
    errors: list[BaseException] = []

    def run(lo: int, hi: int) -> None:
        try:
            work(lo, hi)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
            errors.append(exc)

    threads = []
    for t in range(nt):
        lo = t * chunk
        if lo >= n:
            break
        hi = min(n, lo + chunk)
        th = threading.Thread(target=run, args=(lo, hi))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]


@njit(cache=True, nogil=True)
def _trace_stack_paths_chunk(
    stack: Any,
    origins: np.ndarray,
    dirs: np.ndarray,
    ray_length: float,
    starts: np.ndarray,
    ends: np.ndarray,
    kinds: np.ndarray,
    counts: np.ndarray,
    lo: int,
    hi: int,
) -> None:
    for i in range(lo, hi):
        c, ex, fo, fd = _trace_stack_segments_njit(
            stack, origins[i], dirs[i], ray_length, starts[i], ends[i], kinds[i]
        )
        counts[i] = c


def trace_segments_batch(
    elements: Sequence["RefractiveElement"],
    origins: np.ndarray,
    dirs: np.ndarray,
    ray_length: float = 3.0,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None":
    """Parallel per-ray path tracing for rendering.

    Returns ``(starts, ends, kinds, counts)`` with per-ray segment lists, or
    ``None`` when the stack cannot run in nopython mode.
    """
    ctx = build_trace_context(elements)
    if ctx is None:
        return None
    origins = np.ascontiguousarray(origins, dtype=float)
    dirs = np.ascontiguousarray(dirs, dtype=float)
    n = origins.shape[0]
    cap = _segments_capacity(len(elements))
    starts = np.empty((n, cap, 3), dtype=float)
    ends = np.empty((n, cap, 3), dtype=float)
    kinds = np.empty((n, cap), dtype=np.int8)
    counts = np.zeros(n, dtype=np.int64)

    def work(lo: int, hi: int) -> None:
        _trace_stack_paths_chunk(
            ctx, origins, dirs, float(ray_length), starts, ends, kinds, counts, lo, hi
        )

    _run_in_chunks(work, n)
    return starts, ends, kinds, counts


_SEGMENT_KIND_NAMES = ("incident", "internal", "exit")


def _trace_stack_path_fast(
    ray: Ray, ctx: "tuple[Any, ...]", ray_length: float, n_elements: int
) -> "tuple[list[tuple[np.ndarray, np.ndarray, str]], Ray | None]":
    cap = _segments_capacity(n_elements)
    starts = np.empty((cap, 3), dtype=float)
    ends = np.empty((cap, 3), dtype=float)
    kinds = np.empty(cap, dtype=np.int8)
    count, exited, fo, fd = _trace_stack_segments_njit(
        ctx,
        np.asarray(ray.origin, dtype=float),
        np.asarray(ray.direction, dtype=float),
        float(ray_length),
        starts,
        ends,
        kinds,
    )
    segments = [
        (starts[i].copy(), ends[i].copy(), _SEGMENT_KIND_NAMES[int(kinds[i])])
        for i in range(count)
    ]
    if not exited:
        return segments, None
    return segments, Ray(fo, fd, ray.power)


@dataclass(frozen=True)
class TraceResult:
    incident_hit: RayHit | None
    exit_hit: RayHit | None
    incident_ray: Ray
    internal_path: tuple[Ray, ...]
    outgoing_ray: Ray | None


def _nearest_entry(
    elements: Sequence["RefractiveElement"], ray: Ray, t_max: float = np.inf
) -> "tuple[float, RefractiveElement] | None":
    best: tuple[float, RefractiveElement] | None = None
    for element in elements:
        entry = element.first_hit(ray)
        if entry is None or entry[0] >= t_max:
            continue
        if best is None or entry[0] < best[0]:
            best = (entry[0], element)
    return best


def traverse_stack(
    ray: Ray,
    elements: Sequence["RefractiveElement"],
    limit_fn: "Callable[[Ray], float] | None" = None,
) -> Iterator["tuple[RefractiveElement, TraceResult]"]:
    """Yield ``(element, TraceResult)`` visiting elements in geometric order.

    Iteration stops when no remaining element is entered before
    ``limit_fn(ray)`` (if given), when no element is hit, or when a ray
    is absorbed inside an element. An element already containing the ray
    is visited at its nearest face hit and traced from its interior.
    """
    current = ray
    remaining = list(elements)
    while remaining:
        t_max = np.inf if limit_fn is None else float(limit_fn(current))
        best = _nearest_entry(remaining, current, t_max)
        if best is None:
            return
        remaining.remove(best[1])
        result = best[1].trace(current)
        yield best[1], result
        if result.outgoing_ray is None:
            return
        current = result.outgoing_ray


def internal_rays(result: TraceResult) -> "Iterator[tuple[Ray, np.ndarray | None]]":
    """Yield ``(ray, endpoint)`` for each segment traveled inside an element.

    ``endpoint`` is ``None`` when the internal ray leaves the element's
    interior without a recorded hit (open side or bounce limit reached).
    """
    path = result.internal_path
    for i, segment in enumerate(path):
        if i + 1 < len(path):
            yield segment, path[i + 1].origin
        elif result.exit_hit is not None:
            yield segment, result.exit_hit.point
        else:
            yield segment, None


def trace_stack(ray: Ray, elements: Sequence["RefractiveElement"]) -> Ray | None:
    """Return the ray leaving a stack of elements, or None if stopped.

    Elements are visited in geometric order (nearest entry hit first),
    independent of their order in the stack.
    """
    current: Ray | None = ray
    for _, result in traverse_stack(ray, elements):
        current = result.outgoing_ray
    return current


def trace_stack_path(
    ray: Ray, elements: Sequence["RefractiveElement"], ray_length: float = 3.0
) -> "tuple[list[tuple[np.ndarray, np.ndarray, str]], Ray | None]":
    """Return ``(segments, final_ray)`` for rendering.

    Each segment is a ``(start, end, kind)`` tuple where ``kind`` is
    ``"incident"`` before the first element, ``"internal"`` inside an
    element, and ``"exit"`` after leaving one.

    Runs through the nopython stack kernel when every surface supports it,
    otherwise falls back to the Python traversal below.
    """
    ctx = build_trace_context(elements)
    if ctx is not None:
        return _trace_stack_path_fast(ray, ctx, ray_length, len(elements))
    segments: list[tuple[np.ndarray, np.ndarray, str]] = []
    current = ray
    kind = "incident"
    for _, result in traverse_stack(ray, elements):
        if result.incident_hit is not None:
            segments.append((current.origin, result.incident_hit.point, kind))
        for segment, endpoint in internal_rays(result):
            if endpoint is None:
                continue
            segments.append((segment.origin, endpoint, "internal"))
        if result.outgoing_ray is None:
            return segments, None
        current = result.outgoing_ray
        kind = "exit"
    end = current.origin + ray_length * current.direction
    segments.append((current.origin, end, kind))
    return segments, current

class RefractiveElement:
    surface1: Surface
    surface2: Surface
    material: Material

    def __init__(self, surface1: Surface, surface2: Surface, material: Material) -> None:
        self.surface1 = surface1
        self.surface2 = surface2
        self.material = material
        self._sidewall: Any = _SIDEWALL_UNSET
        self._local_aabb: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def transform(self) -> Transform:
        return self.surface1.transform

    @property
    def sidewall(self) -> LensSidewall | None:
        if self._sidewall is _SIDEWALL_UNSET:
            if (
                self.surface1.aperture is None
                or self.surface2.aperture is None
            ):
                self._sidewall = None
            else:
                front = rear = None

                try:
                    front = self.surface1.aperture.boundary(self.surface1)
                    rear = self.surface2.aperture.boundary(self.surface2)
                except ValueError:
                    front = rear = None

                if (
                    front is not None
                    and rear is not None
                    and front.shape == rear.shape
                    and not np.allclose(front, rear, atol=_c.SIDEWALL_MATCH_ATOL)
                ):
                    self._sidewall = LensSidewall(front, rear)
                else:
                    self._sidewall = None
        return self._sidewall

    def _local_bounds(self) -> "tuple[np.ndarray, np.ndarray]":
        """Cached axis-aligned bounds of both surfaces in the local frame."""
        if self._local_aabb is None:
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            for surface in (self.surface1, self.surface2):
                (u_min, u_max), (v_min, v_max) = surface.parameter_range
                us = np.linspace(u_min, u_max, _c.CONTAINMENT_BOUNDS_N)
                vs = np.linspace(v_min, v_max, _c.CONTAINMENT_BOUNDS_N)
                U, V = np.meshgrid(us, vs)
                pts = surface.evaluate_local(U, V).reshape(-1, 3)
                pts = pts[np.all(np.isfinite(pts), axis=1)]
                if pts.size == 0:
                    continue
                lo = np.minimum(lo, pts.min(axis=0))
                hi = np.maximum(hi, pts.max(axis=0))
            self._local_aabb = (lo, hi)
        return self._local_aabb

    def _local_surface_z(self, surface: Surface, local: np.ndarray) -> float | None:
        """Local height of ``surface`` above ``(local[0], local[1])``."""
        u = float(local[0])
        v = float(local[1])
        if surface.is_graph_over_xy:
            if not surface.accepts_parameters(u, v):
                return None
        else:
            uv = surface.parameters_at_xy(u, v)
            if uv is None:
                return None
            u, v = uv
        z = float(surface.evaluate_local(u, v)[2])
        return z if np.isfinite(z) else None

    def contains_point(self, p: np.ndarray | Sequence[float]) -> bool:
        """Whether ``p`` lies inside the volume enclosed by this element.

        The volume is the region between the two surfaces over their shared
        aperture, expressed in the element's local frame; this matches the
        graph-over-XY assumption already made by the sidewall and meshing
        code. Points outside the aperture, or on elements whose surfaces are
        not graphs over local XY, report ``False``.
        """
        local = self.transform.point_to_local(np.asarray(p, dtype=float))
        if not np.all(np.isfinite(local)):
            return False
        lo, hi = self._local_bounds()
        if np.any(local < lo) or np.any(local > hi):
            return False
        z1 = self._local_surface_z(self.surface1, local)
        z2 = self._local_surface_z(self.surface2, local)
        if z1 is None or z2 is None:
            return False
        return bool(min(z1, z2) < local[2] < max(z1, z2))

    def _candidate_hits(
        self, ray: Ray, min_t: float = 0.0
    ) -> "tuple[float, RayHit, np.ndarray] | None":
        hits: list[tuple[float, RayHit, np.ndarray]] = []
        for surface in (self.surface1, self.surface2):
            surface_hit = surface.intersect(ray)
            if surface_hit is not None and surface_hit.t > min_t:
                normal = surface.normal(surface_hit.u, surface_hit.v)
                if normal is not None:
                    hits.append((surface_hit.t, surface_hit, np.asarray(normal, dtype=float)))
        wall = self.sidewall
        if wall is not None:
            wall_hit = wall.intersect(ray)
            if wall_hit is not None and wall_hit.t > min_t:
                hits.append((wall_hit.t, wall_hit, wall.normal(wall_hit.face)))
        if not hits:
            return None
        return min(hits, key=lambda item: item[0])

    def first_hit(self, ray: Ray) -> "tuple[float, RayHit, np.ndarray] | None":
        """Nearest intersection with any face of the element.

        Returns ``(t, hit, normal)`` or None.
        """
        return self._candidate_hits(ray)

    def trace(self, ray: Ray) -> TraceResult:
        """Trace a ray through the closed element (surfaces + sidewall).

        A ray starting outside the element refracts into the glass at its
        nearest face, bounces off interior faces on total internal
        reflection (up to ``_MAX_INTERNAL_BOUNCES``), and refracts out when
        it reaches a subcritical hit.

        A ray originating inside the element is traced from that medium
        instead: it refracts out (or reflects, on total internal
        reflection) at its first face hit and reports ``incident_hit=None``.
        """
        inside = self.contains_point(ray.origin)
        entry = self.first_hit(ray)
        if entry is None:
            if inside:
                # Originating inside an open element with no face ahead:
                # the ray leaves through the opening undeviated.
                return TraceResult(None, None, ray, (ray,), ray)
            return TraceResult(None, None, ray, (), None)

        n = self.material.refractive_index
        _, entry_hit, entry_normal = entry

        path: list[Ray] = []
        incident_hit: RayHit | None = None
        if inside:
            current = ray
        else:
            incident_hit = entry_hit
            d, _ = refract_or_reflect(
                ray.direction, _orient(entry_normal, ray.direction), 1.0, n
            )
            current = Ray(entry_hit.point + _EPSILON * d, d, ray.power)

        for _ in range(_MAX_INTERNAL_BOUNCES):
            internal = self._candidate_hits(current, min_t=_MIN_HIT_T)
            if internal is None:
                path.append(current)
                return TraceResult(incident_hit, None, ray, tuple(path), None)

            _, face_hit, face_normal = internal
            path.append(current)
            d, refracted = refract_or_reflect(
                current.direction, _orient(face_normal, current.direction), n, 1.0
            )
            if refracted:
                return TraceResult(
                    incident_hit,
                    face_hit,
                    ray,
                    tuple(path),
                    Ray(face_hit.point, d, ray.power),
                )
            current = Ray(face_hit.point + _EPSILON * d, d, ray.power)

        path.append(current)
        return TraceResult(incident_hit, None, ray, tuple(path), None)
