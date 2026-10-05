import numpy as np
import json

from .surface import Surface
from .sidewall import LensSidewall

_SIDEWALL_UNSET = object()

class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)

def refract_direction(incident, normal, n_from, n_to):
    eta = n_from / n_to
    cos_i = float(np.dot(normal, incident))
    k = 1.0 - eta**2 * (1.0 - cos_i**2)
    if k < 0.0:
        return incident - 2.0 * cos_i * normal
    return eta * incident + (np.sqrt(k) - eta * cos_i) * normal

class RefractiveElement:
    def __init__(self, schema: str, orientation=np.array([0, 0, 1])):
        self.sch = json.loads(schema)
        self.n = self.sch["material"]["refractive_index"]
        self.orientation = orientation / np.linalg.norm(orientation)
        original = np.array([0, 0, 1])
        new = self.orientation
        axis = np.cross(original, new)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            if np.inner(original, new) < 0:
                self.r_matrix = np.array([[1.0, 0.0, 0.0],
                                          [0.0, -1.0, 0.0],
                                          [0.0, 0.0, -1.0]])
            else:
                self.r_matrix = np.eye(3, 3)
        else:
            axis_mat = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
            self.r_matrix = np.eye(3, 3) + axis_mat + np.matmul(axis_mat, axis_mat) * ((1 - np.inner(original, new)) / axis_norm**2)
        self.surface1 = self._make_surface(self.sch["surface1"], self.r_matrix)
        self.surface2 = self._make_surface(self.sch["surface2"], self.r_matrix)
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

        p1, u1, v1 = hit1
        n1 = self.surface1.normal(u1, v1)

        if n1 is None:
            return None

        d1 = refract_direction(ray.direction, n1, n_from=1.0, n_to=self.n)
        r1 = Ray(p1, d1)

        hit2 = self.surface2.intersect(r1)

        if hit2 is None:
            return None

        p2, u2, v2 = hit2
        n2 = self.surface2.normal(u2, v2)

        if n2 is None:
            return None

        d2 = refract_direction(d1, n2, n_from=self.n, n_to=1.0)

        return Ray(p2, d2)

def check_on_axis(label, element, orientation):
    s = orientation / np.linalg.norm(orientation)
    ray = Ray(2.0 * s, -s)

    hit = element.surface1.intersect(ray)
    p1 = hit[0] if hit is not None else None
    assert p1 is not None, f"{label}: no entry intersection"
    assert np.isclose(np.linalg.norm(p1), 1.0, atol=1e-6), f"{label}: entry point not on sphere: {p1}"
    assert (element.r_matrix.T @ p1)[2] > 0.0, f"{label}: entry point not on surface1 hemisphere: {p1}"

    out = element.refract(ray)
    assert out.origin is not None, f"{label}: no exit point"
    assert np.allclose(out.direction, -s, atol=1e-6), f"{label}: on-axis ray deflected: {out.direction}"
    assert np.isclose(np.linalg.norm(out.origin), 1.0, atol=1e-6), f"{label}: exit point not on sphere: {out.origin}"

    print(f"{label}: origin={out.origin}, direction={out.direction}")


if __name__ == "__main__":
    with open("tests/test_geometry.json", "r") as f:
        schema = f.read()

    check_on_axis("default orientation", RefractiveElement(schema), np.array([0.0, 0.0, 1.0]))
    check_on_axis("tilted orientation", RefractiveElement(schema, np.array([1.0, 0.5, 1.0])), np.array([1.0, 0.5, 1.0]))
