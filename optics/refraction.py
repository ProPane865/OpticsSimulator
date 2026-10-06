import numpy as np
import json

from .surface import Surface
from .sidewall import LensSidewall
from .transform import Transform
from .vector import normalize, rotation_from_z
from .ray import Ray

_SIDEWALL_UNSET = object()

def refract_direction(incident, normal, n_from, n_to):
    eta = n_from / n_to
    cos_i = float(np.dot(normal, incident))
    k = 1.0 - eta**2 * (1.0 - cos_i**2)
    if k < 0.0:
        return incident - 2.0 * cos_i * normal
    return eta * incident + (np.sqrt(k) - eta * cos_i) * normal

class RefractiveElement:
    def __init__(self, schema: str, orientation=np.array([0, 0, 1]), position=None):
        self.sch = json.loads(schema)
        self.n = self.sch["material"]["refractive_index"]
        position = np.zeros(3) if position is None else np.asarray(position, dtype=float)
        self.transform = Transform(rotation_from_z(orientation), position)

        self.surface1 = self._make_surface(self.sch["surface1"], self.transform)
        self.surface2 = self._make_surface(self.sch["surface2"], self.transform)
        self._sidewall = _SIDEWALL_UNSET

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

    @staticmethod
    def _make_surface(entry, transform):
        return Surface(
            entry["x"],
            entry["y"],
            entry["z"],
            transform=transform,
            u_range=entry.get("u_range", (-2.0, 2.0)),
            v_range=entry.get("v_range", (-2.0, 2.0)),
            aperture=entry.get("aperture")
        )

    def refract(self, ray: Ray) -> Ray:
        hit1 = self.surface1.intersect(ray)

        if hit1 is None:
            return None

        p1, u1, v1 = hit1.point, hit1.u, hit1.v
        n1 = self.surface1.normal(u1, v1)

        if n1 is None:
            return None

        d1 = refract_direction(ray.direction, n1, n_from=1.0, n_to=self.n)
        r1 = Ray(p1, d1)

        hit2 = self.surface2.intersect(r1)

        if hit2 is None:
            return None

        p2, u2, v2 = hit2.point, hit2.u, hit2.v
        n2 = self.surface2.normal(u2, v2)

        if n2 is None:
            return None

        d2 = refract_direction(d1, n2, n_from=self.n, n_to=1.0)

        return Ray(p2, d2)
