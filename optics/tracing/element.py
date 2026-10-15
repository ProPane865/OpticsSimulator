"""Closed refractive element: geometry, containment, face hits, Python tracing."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import constants as _c
from ..primitive import LensSidewall, Material, Ray, RayHit, Surface, Transform
from .direction import _orient, refract_or_reflect

_SIDEWALL_UNSET = object()

_EPSILON = _c.RAY_EPSILON
_MIN_HIT_T = _c.MIN_HIT_T
_MAX_INTERNAL_BOUNCES = _c.MAX_INTERNAL_BOUNCES


@dataclass(frozen=True)
class TraceResult:
    incident_hit: RayHit | None
    exit_hit: RayHit | None
    incident_ray: Ray
    internal_path: tuple[Ray, ...]
    outgoing_ray: Ray | None


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

    def local_bounds(self) -> "tuple[np.ndarray, np.ndarray]":
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
        lo, hi = self.local_bounds()
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
