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
        assert result.incident_hit is not None
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
    surface1: Surface
    surface2: Surface
    material: Material

    def __init__(self, surface1: Surface, surface2: Surface, material: Material) -> None:
        self.surface1 = surface1
        self.surface2 = surface2
        self.material = material
        self._sidewall: Any = _SIDEWALL_UNSET

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

    def _candidate_hits(
        self, ray: Ray, min_t: float = 0.0
    ) -> "tuple[float, RayHit, np.ndarray] | None":
        hits: list[tuple[float, RayHit, np.ndarray]] = []
        for surface in (self.surface1, self.surface2):
            surface_hit = surface.intersect(ray)
            if (
                surface_hit is not None
                and surface_hit.t > min_t
                and surface_hit.u is not None
                and surface_hit.v is not None
            ):
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

        path: list[Ray] = []
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
