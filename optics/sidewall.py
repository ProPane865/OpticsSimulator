import numpy as np

from .ray import Ray
from .hit import Hit

class LensSidewall:
    def __init__(self, front, rear):
        self.front = np.array(front, dtype=float)
        self.rear = np.array(rear, dtype=float)

        if self.front.shape != self.rear.shape:
            raise ValueError("Front and rear rims must correspond")

        if self.front.ndim != 2 or self.front.shape[1] != 3:
            raise ValueError("Rims must have shape (N, 3)")

        if len(self.front) < 3:
            raise ValueError("Rims must have at least 3 points")

        if not (
            np.all(np.isfinite(self.front))
            and np.all(np.isfinite(self.rear))
        ):
            raise ValueError("Rim points must be finite")

        n = len(self.front)
        self._verts = np.vstack([self.front, self.rear])

        i = np.arange(n)
        j = (i + 1) % n
        f0, f1 = i, j
        r0, r1 = n + i, n + j

        self._faces = np.empty((2 * n, 3), dtype=np.int64)
        self._faces[0::2] = np.column_stack([f0, r0, r1])
        self._faces[1::2] = np.column_stack([f0, r1, f1])

    def mesh(self):
        return self._verts, self._faces

    def intersect(self, ray):
        o = np.asarray(ray.origin, dtype=float)
        d = np.asarray(ray.direction, dtype=float)

        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
            return None

        ray = Ray(o, d)

        tri = self._verts[self._faces]
        a = tri[:, 0]
        e1 = tri[:, 1] - a
        e2 = tri[:, 2] - a

        h = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, h)

        candidate = np.abs(det) > 1e-14

        inv_det = np.zeros_like(det)
        inv_det[candidate] = 1.0 / det[candidate]

        s = o - a
        u = np.einsum("ij,ij->i", s, h) * inv_det
        q = np.cross(s, e1)
        v = (q @ d) * inv_det
        t = np.einsum("ij,ij->i", e2, q) * inv_det

        tol = 1e-12

        hit = (
            candidate
            & (t > 1e-9)
            & (u >= -tol)
            & (v >= -tol)
            & (u + v <= 1.0 + tol)
        )

        if not hit.any():
            return None

        k = int(np.argmin(np.where(hit, t, np.inf)))

        u_k = min(max(float(u[k]), 0.0), 1.0)
        v_k = min(max(float(v[k]), 0.0), 1.0 - u_k)

        return Hit(ray.at(float(t[k])), float(t[k]), u_k, v_k)