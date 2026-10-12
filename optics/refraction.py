from dataclasses import dataclass

import numpy as np

from . import constants as _c
from .hit import RayHit
from .material import Material
from .ray import Ray
from .sidewall import LensSidewall
from .surface import Surface
from .vector import normalize

_SIDEWALL_UNSET = object()

_EPSILON = _c.RAY_EPSILON
_MIN_HIT_T = _c.MIN_HIT_T
_MAX_INTERNAL_BOUNCES = _c.MAX_INTERNAL_BOUNCES

def _reflect_direction(incident, normal):
    return normalize(incident - 2.0 * float(np.dot(normal, incident)) * normal)

def _refract_direction(incident, normal, eta, cos_i, k):
    return normalize(eta * incident + (np.sqrt(k) - eta * cos_i) * normal)

def _orient(normal, incident):
    """Align the surface normal with the incident ray direction."""
    normal = np.asarray(normal, dtype=float)
    if float(np.dot(normal, incident)) < 0.0:
        return -normal
    return normal

def refract_or_reflect(incident, normal, n_from, n_to):
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

def trace_direction(incident, normal, n_from, n_to):
    return refract_or_reflect(incident, normal, n_from, n_to)[0]

@dataclass(frozen=True)
class TraceResult:
    incident_hit: RayHit | None
    exit_hit: RayHit | None
    incident_ray: Ray
    internal_path: tuple[Ray, ...]
    outgoing_ray: Ray | None


def _nearest_entry(elements, ray, t_max=np.inf):
    best = None
    for element in elements:
        entry = element.first_hit(ray)
        if entry is None or entry[0] >= t_max:
            continue
        if best is None or entry[0] < best[0]:
            best = (entry[0], element)
    return best


def traverse_stack(ray, elements, limit_fn=None):
    """Yield ``(element, TraceResult)`` visiting elements in geometric order.

    Iteration stops when no remaining element is entered before
    ``limit_fn(ray)`` (if given), when no element is hit, or when a ray
    is absorbed inside an element.
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


def trace_stack(ray, elements):
    """Return the ray leaving a stack of elements, or None if stopped.

    Elements are visited in geometric order (nearest entry hit first),
    independent of their order in the stack.
    """
    current = ray
    for _, result in traverse_stack(ray, elements):
        current = result.outgoing_ray
    return current


def trace_stack_path(ray, elements, ray_length=3.0):
    """Return ``(segments, final_ray)`` for rendering.

    Each segment is a ``(start, end, kind)`` tuple where ``kind`` is
    ``"incident"`` before the first element, ``"internal"`` inside an
    element, and ``"exit"`` after leaving one.
    """
    segments = []
    current = ray
    kind = "incident"
    for _, result in traverse_stack(ray, elements):
        segments.append((current.origin, result.incident_hit.point, kind))
        path = result.internal_path
        for i, segment in enumerate(path):
            if i + 1 < len(path):
                end = path[i + 1].origin
            elif result.exit_hit is not None:
                end = result.exit_hit.point
            else:
                break
            segments.append((segment.origin, end, "internal"))
        if result.outgoing_ray is None:
            return segments, None
        current = result.outgoing_ray
        kind = "exit"
    end = current.origin + ray_length * current.direction
    segments.append((current.origin, end, kind))
    return segments, current

class RefractiveElement:
    def __init__(self, surface1: Surface, surface2: Surface, material: Material):
        self.surface1 = surface1
        self.surface2 = surface2
        self.material = material
        self._sidewall = _SIDEWALL_UNSET

    @property
    def transform(self):
        return self.surface1.transform

    @property
    def sidewall(self):
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

    def _candidate_hits(self, ray, min_t=0.0):
        hits = []
        for surface in (self.surface1, self.surface2):
            hit = surface.intersect(ray)
            if hit is not None and hit.t > min_t:
                normal = surface.normal(hit.u, hit.v)
                if normal is not None:
                    hits.append((hit.t, hit, np.asarray(normal, dtype=float)))
        wall = self.sidewall
        if wall is not None:
            hit = wall.intersect(ray)
            if hit is not None and hit.t > min_t:
                hits.append((hit.t, hit, wall.normal(hit.face)))
        if not hits:
            return None
        return min(hits, key=lambda item: item[0])

    def first_hit(self, ray):
        """Nearest intersection with any face of the element.

        Returns ``(t, hit, normal)`` or None.
        """
        return self._candidate_hits(ray)

    def trace(self, ray) -> TraceResult:
        """Trace a ray through the closed element (surfaces + sidewall).

        The ray refracts into the glass at its nearest face, bounces off
        interior faces on total internal reflection (up to
        ``_MAX_INTERNAL_BOUNCES``), and refracts out when it reaches a
        subcritical hit.
        """
        entry = self.first_hit(ray)
        if entry is None:
            return TraceResult(None, None, ray, (), None)

        n = self.material.refractive_index
        _, entry_hit, entry_normal = entry
        d, _ = refract_or_reflect(
            ray.direction, _orient(entry_normal, ray.direction), 1.0, n
        )

        path = []
        current = Ray(entry_hit.point + _EPSILON * d, d, ray.power)
        for _ in range(_MAX_INTERNAL_BOUNCES):
            internal = self._candidate_hits(current, min_t=_MIN_HIT_T)
            if internal is None:
                path.append(current)
                return TraceResult(entry_hit, None, ray, tuple(path), None)

            _, face_hit, face_normal = internal
            path.append(current)
            d, refracted = refract_or_reflect(
                current.direction, _orient(face_normal, current.direction), n, 1.0
            )
            if refracted:
                return TraceResult(
                    entry_hit,
                    face_hit,
                    ray,
                    tuple(path),
                    Ray(face_hit.point, d, ray.power),
                )
            current = Ray(face_hit.point + _EPSILON * d, d, ray.power)

        path.append(current)
        return TraceResult(entry_hit, None, ray, tuple(path), None)
