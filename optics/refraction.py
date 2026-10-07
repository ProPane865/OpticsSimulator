import numpy as np

from .surface import Surface
from .sidewall import LensSidewall
from .material import Material
from .hit import Hit
from .ray import Ray

from dataclasses import dataclass

_SIDEWALL_UNSET = object()

def _refract_direction(incident, normal, eta, cos_i, k):
    return eta * incident + (np.sqrt(k) - eta * cos_i) * normal

def _reflect_direction(incident, normal, cos_i):
    return incident - 2.0 * cos_i * normal

def trace_direction(incident, normal, n_from, n_to):
    eta = n_from / n_to
    cos_i = float(np.dot(normal, incident))
    k = 1.0 - eta**2 * (1.0 - cos_i**2)

    if k < 0.0:
        return _reflect_direction(incident, normal, cos_i)
    return _refract_direction(incident, normal, eta, cos_i, k)

@dataclass(frozen=True)
class TraceResult:
    incident_hit: Hit | None
    exit_hit: Hit | None
    incident_ray: Ray
    internal_ray: Ray | None
    outgoing_ray: Ray | None

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
                    and not np.allclose(front, rear, atol=1e-6)
                ):
                    self._sidewall = LensSidewall(front, rear)
                else:
                    self._sidewall = None
        return self._sidewall

    def trace(self, ray) -> TraceResult:
        hit1 = self.surface1.intersect(ray)

        if hit1 is None:
            return TraceResult(None, None, ray, None, None)

        p1, u1, v1 = hit1.point, hit1.u, hit1.v
        n1 = self.surface1.normal(u1, v1)

        if n1 is None:
            return TraceResult(hit1, None, ray, None, None)

        d1 = trace_direction(ray.direction, n1, n_from=1.0, n_to=self.material.refractive_index)
        r1 = Ray(p1, d1)

        hit2 = self.surface2.intersect(r1)

        if hit2 is None:
            return TraceResult(hit1, None, ray, r1, None)

        p2, u2, v2 = hit2.point, hit2.u, hit2.v
        n2 = self.surface2.normal(u2, v2)

        if n2 is None:
            return TraceResult(hit1, hit2, ray, r1, None)

        d2 = trace_direction(d1, n2, n_from=self.material.refractive_index, n_to=1.0)
        r2 = Ray(p2, d2)

        return TraceResult(hit1, hit2, ray, r1, r2)
