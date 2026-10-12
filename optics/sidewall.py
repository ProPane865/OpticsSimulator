import numpy as np
from numba import njit

from . import constants as _c
from .hit import MeshHit
from .ray import Ray


@njit(cache=True)
def _mt_kernel(
    o: np.ndarray,
    d: np.ndarray,
    tri: np.ndarray,
    det_eps: float,
    bary_tol: float,
    min_face_t: float,
) -> "tuple[float, int, float, float, bool]":
    """Moller-Trumbore over ``tri`` (N, 3, 3); nearest forward hit.

    Returns ``(t, face, u, v, found)`` with barycentric ``u``/``v`` clamped
    into the triangle.
    """
    best_t = np.inf
    best_face = -1
    best_u = 0.0
    best_v = 0.0
    found = False

    for k in range(tri.shape[0]):
        a = tri[k, 0]
        e1 = tri[k, 1] - a
        e2 = tri[k, 2] - a

        h = np.cross(d, e2)
        det = float(np.dot(e1, h))
        if abs(det) <= det_eps:
            continue
        inv_det = 1.0 / det

        s = o - a
        u = float(np.dot(s, h)) * inv_det
        q = np.cross(s, e1)
        v = float(np.dot(q, d)) * inv_det
        t = float(np.dot(e2, q)) * inv_det

        if (
            t > min_face_t
            and u >= -bary_tol
            and v >= -bary_tol
            and u + v <= 1.0 + bary_tol
            and t < best_t
        ):
            best_t = t
            best_face = k
            best_u = u
            best_v = v
            found = True

    if not found:
        return 0.0, -1, 0.0, 0.0, False

    u_k = min(max(best_u, 0.0), 1.0)
    v_k = min(max(best_v, 0.0), 1.0 - u_k)
    return best_t, best_face, u_k, v_k, True


class LensSidewall:
    front: np.ndarray
    rear: np.ndarray

    def __init__(self, front: np.ndarray, rear: np.ndarray) -> None:
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

        self._tri = self._verts[self._faces]
        e1 = self._tri[:, 1] - self._tri[:, 0]
        e2 = self._tri[:, 2] - self._tri[:, 0]
        normals = np.cross(e1, e2)
        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0.0] = 1.0
        self._face_normals = normals / norms[:, None]

    def mesh(self) -> "tuple[np.ndarray, np.ndarray]":
        return self._verts, self._faces

    def normal(self, face: int) -> np.ndarray:
        """Outward unit normal of the triangle with the given face index."""
        return self._face_normals[int(face)]

    def intersect(self, ray: Ray) -> MeshHit | None:
        o = np.asarray(ray.origin, dtype=float)
        d = np.asarray(ray.direction, dtype=float)

        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
            return None

        ray = Ray(o, d, ray.power)

        t, face, u_k, v_k, found = _mt_kernel(
            o, d, self._tri, _c.DET_SINGULAR_EPS, _c.BARY_TOL, _c.MIN_FACE_T
        )
        if not found:
            return None

        return MeshHit(
            ray.at(t),
            t,
            int(face),
            np.array([1.0 - u_k - v_k, u_k, v_k], dtype=float),
        )