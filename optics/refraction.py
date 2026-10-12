from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import constants as _c
from .hit import RayHit
from .material import Material
from .ray import Ray
from .sidewall import LensSidewall
from .surface import Surface
from .transform import Transform
from .vector import normalize

_SIDEWALL_UNSET = object()

_EPSILON = _c.RAY_EPSILON
_MIN_HIT_T = _c.MIN_HIT_T
_MAX_INTERNAL_BOUNCES = _c.MAX_INTERNAL_BOUNCES


def _reflect_direction(incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return normalize(incident - 2.0 * float(np.dot(normal, incident)) * normal)


def _refract_direction(
    incident: np.ndarray, normal: np.ndarray, eta: float, cos_i: float, k: float
) -> np.ndarray:
    return normalize(eta * incident + (np.sqrt(k) - eta * cos_i) * normal)


def _orient(normal: np.ndarray, incident: np.ndarray) -> np.ndarray:
    """Align the surface normal with the incident ray direction."""
    normal = np.asarray(normal, dtype=float)
    if float(np.dot(normal, incident)) < 0.0:
        return -normal
    return normal


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


def trace_direction(incident: np.ndarray, normal: np.ndarray, n_from: float, n_to: float) -> np.ndarray:
    return refract_or_reflect(incident, normal, n_from, n_to)[0]

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
    """
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
