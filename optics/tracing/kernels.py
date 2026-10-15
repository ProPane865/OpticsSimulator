"""Nopython kernels: element first-hit, containment, and stack walking.

Element context tuple layout (and its index constants) live in
``optics.numerical._kernels``.
"""

import math
from typing import Any

import numpy as np
from numba import njit

from .. import constants as _c
from ..numerical._kernels import (
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
)
from ..numerical.desc import accepts_njit, derivatives_njit, eval_njit
from ..numerical.intersect import surface_intersect_njit
from ..numerical.invert import parameters_at_xy_njit
from ..numerical.linalg import norm
from ..numerical.mesh import mesh_intersect_kernel
from ..numerical.plane import plane_intersect_kernel
from .direction import _orient, refract_or_reflect

_MAX_INTERNAL_BOUNCES = _c.MAX_INTERNAL_BOUNCES
_MAX_PATH_PTS = _MAX_INTERNAL_BOUNCES + 2
_MAX_SEGS_PER_ELEMENT = 2 * _MAX_INTERNAL_BOUNCES + 3
_MIN_HIT_T = _c.MIN_HIT_T
_EPSILON = _c.RAY_EPSILON
_DET_SINGULAR_EPS = _c.DET_SINGULAR_EPS
_BARY_TOL = _c.BARY_TOL
_MIN_FACE_T = _c.MIN_FACE_T


@njit(cache=True)
def _normal_njit(desc: Any, u: float, v: float) -> "tuple[float, float, float, bool]":
    """World-space unit normal at (u, v); mirrors ``Surface.normal``."""
    if not accepts_njit(desc, u, v):
        return 0.0, 0.0, 0.0, False
    ru, rv = derivatives_njit(desc, u, v)
    n = -np.cross(ru, rv)
    norm_len = norm(n)
    if not math.isfinite(norm_len) or norm_len <= 0.0:
        return 0.0, 0.0, 0.0, False
    w = desc[D_ROT] @ (n / norm_len)
    return float(w[0]), float(w[1]), float(w[2]), True


@njit(cache=True)
def _surface_hit_njit(
    desc: Any, grid: Any, o: np.ndarray, d: np.ndarray, min_t: float
) -> "tuple[bool, float, np.ndarray, np.ndarray]":
    """Nearest surface hit with normal, or ``(False, inf, 0, 0)``."""
    t, u, v, ok = surface_intersect_njit(desc, grid, o, d)
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
    """Nearest face hit (surfaces + sidewall); mirrors ``RefractiveElement.first_hit``."""
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
        t, face, bu, bv, ok = mesh_intersect_kernel(
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
        if not accepts_njit(desc, u, v):
            return math.nan
    else:
        uu, vv, ok = parameters_at_xy_njit(desc, grid, u, v)
        if not ok:
            return math.nan
        u = uu
        v = vv
    z = float(eval_njit(desc, u, v)[2])
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
def _nearest_unvisited(
    stack: Any, visited: np.ndarray, o: np.ndarray, d: np.ndarray, t_limit: float
) -> int:
    """Index of the nearest unvisited element entered before ``t_limit``, or -1."""
    best_i = -1
    best_t = np.inf
    i = 0
    for elem in stack:
        if not visited[i]:
            found, t, hp, hn = _element_first_hit(elem, o, d, 0.0)
            if found and t < t_limit and t < best_t:
                best_t = t
                best_i = i
        i += 1
    return best_i


@njit(cache=True)
def _trace_chosen(
    stack: Any, best_i: int, o: np.ndarray, d: np.ndarray, pts: np.ndarray
) -> "tuple[np.ndarray, np.ndarray, bool, int, bool, np.ndarray]":
    """Trace the element at ``best_i``; the stack tuple is iterated statically.

    Returns ``(exit_o, exit_d, exited, n_pts, has_incident, incident_pt)``.
    """
    exit_o = o
    exit_d = d
    exited = False
    n_pts = 0
    has_inc = False
    ipt = o
    i = 0
    for elem in stack:
        if i == best_i:
            exit_o, exit_d, exited, n_pts, has_inc, ipt = _trace_element_njit(
                elem, o, d, pts
            )
        i += 1
    return exit_o, exit_d, exited, n_pts, has_inc, ipt


@njit(cache=True)
def trace_stack_segments_njit(
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
        best_i = _nearest_unvisited(stack, visited, cur_o, cur_d, np.inf)
        if best_i < 0:
            break
        visited[best_i] = True
        exit_o, exit_d, exited, n_pts, has_inc, ipt = _trace_chosen(
            stack, best_i, cur_o, cur_d, pts
        )
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
def trace_stack_to_plane_njit(
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
        _, _, t_max, ok = plane_intersect_kernel(cur_o, cur_d, po, pn, eps)
        if not ok:
            break
        best_i = _nearest_unvisited(stack, visited, cur_o, cur_d, t_max)
        if best_i < 0:
            break
        visited[best_i] = True
        exit_o, exit_d, exited, n_pts, _, _ = _trace_chosen(
            stack, best_i, cur_o, cur_d, pts
        )
        for k in range(n_pts - 1):
            seg_o = pts[k]
            diff = pts[k + 1] - seg_o
            seg_len = norm(diff)
            if seg_len <= 0.0:
                continue
            seg_d = diff / seg_len
            u, v, t, pok = plane_intersect_kernel(seg_o, seg_d, po, pn, eps)
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
