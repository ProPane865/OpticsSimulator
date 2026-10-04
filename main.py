import numpy as np
import sympy as sp
import json

class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)

class Surface:
    def __init__(
        self,
        x_expr,
        y_expr,
        z_expr,
        transform=None,
        u_range=(-2.0, 2.0),
        v_range=(-2.0, 2.0),
    ):
        self.u = sp.symbols("u")
        self.v = sp.symbols("v")
        self.x_expr = sp.parse_expr(str(x_expr))
        self.y_expr = sp.parse_expr(str(y_expr))
        self.z_expr = sp.parse_expr(str(z_expr))
        self.transform = np.eye(3, 3) if transform is None else np.asarray(transform, dtype=float)
        self.u_range = (float(u_range[0]), float(u_range[1]))
        self.v_range = (float(v_range[0]), float(v_range[1]))
        self._x = sp.lambdify((self.u, self.v), self.x_expr, modules=["numpy"])
        self._y = sp.lambdify((self.u, self.v), self.y_expr, modules=["numpy"])
        self._z = sp.lambdify((self.u, self.v), self.z_expr, modules=["numpy"])
        self._rx = sp.lambdify((self.u, self.v), sp.diff(self.x_expr, self.u), modules=["numpy"])
        self._ry = sp.lambdify((self.u, self.v), sp.diff(self.y_expr, self.u), modules=["numpy"])
        self._rz = sp.lambdify((self.u, self.v), sp.diff(self.z_expr, self.u), modules=["numpy"])
        self._sx = sp.lambdify((self.u, self.v), sp.diff(self.x_expr, self.v), modules=["numpy"])
        self._sy = sp.lambdify((self.u, self.v), sp.diff(self.y_expr, self.v), modules=["numpy"])
        self._sz = sp.lambdify((self.u, self.v), sp.diff(self.z_expr, self.v), modules=["numpy"])

    @property
    def parameter_range(self):
        return (self.u_range, self.v_range)

    def _point_local(self, u, v):
        u = float(u)
        v = float(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            x = float(self._x(u, v))
            y = float(self._y(u, v))
            z = float(self._z(u, v))
        return np.array([x, y, z], dtype=float)

    def _point_local_array(self, U, V):
        shape = np.shape(U)
        with np.errstate(invalid="ignore", divide="ignore"):
            X = np.asarray(self._x(U, V), dtype=float)
            Y = np.asarray(self._y(U, V), dtype=float)
            Z = np.asarray(self._z(U, V), dtype=float)
        return (
            np.broadcast_to(X, shape),
            np.broadcast_to(Y, shape),
            np.broadcast_to(Z, shape),
        )

    def _tangent_local(self, u, v):
        u = float(u)
        v = float(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            ru = np.array(
                [
                    float(self._rx(u, v)),
                    float(self._ry(u, v)),
                    float(self._rz(u, v)),
                ],
                dtype=float,
            )
            rv = np.array(
                [
                    float(self._sx(u, v)),
                    float(self._sy(u, v)),
                    float(self._sz(u, v)),
                ],
                dtype=float,
            )
        return ru, rv

    def point(self, u, v):
        return self.transform @ self._point_local(u, v)

    def tangents(self, u, v):
        ru, rv = self._tangent_local(u, v)
        return self.transform @ ru, self.transform @ rv

    def normal(self, u, v):
        ru, rv = self._tangent_local(u, v)
        n = -np.cross(ru, rv)
        norm = np.linalg.norm(n)
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        return self.transform @ (n / norm)

    def _parameters_in_range(self, u, v):
        tu = 1e-9 * max(1.0, abs(self.u_range[1] - self.u_range[0]))
        tv = 1e-9 * max(1.0, abs(self.v_range[1] - self.v_range[0]))
        return (
            self.u_range[0] - tu <= u <= self.u_range[1] + tu
            and self.v_range[0] - tv <= v <= self.v_range[1] + tv
        )

    def _solve_parameters_point(self, p, u, v):
        for _ in range(50):
            puv = self._point_local(u, v)
            F = puv - p
            if not np.all(np.isfinite(F)):
                return None
            f_norm = float(np.linalg.norm(F))
            if f_norm < 1e-11:
                if self._parameters_in_range(u, v):
                    return float(u), float(v)
                return None
            ru, rv = self._tangent_local(u, v)
            if not (np.all(np.isfinite(ru)) and np.all(np.isfinite(rv))):
                return None
            J = np.column_stack((ru, rv))
            try:
                delta = np.linalg.lstsq(J, -F, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(delta)):
                return None
            u += float(delta[0])
            v += float(delta[1])
            if not (np.isfinite(u) and np.isfinite(v)):
                return None
            step_norm = float(np.linalg.norm(delta))
            if step_norm < 1e-14 * max(1.0, abs(u), abs(v)):
                puv = self._point_local(u, v)
                F = puv - p
                if (
                    np.all(np.isfinite(F))
                    and float(np.linalg.norm(F)) < 1e-9
                    and self._parameters_in_range(u, v)
                ):
                    return float(u), float(v)
                return None
        puv = self._point_local(u, v)
        F = puv - p
        if (
            np.all(np.isfinite(F))
            and float(np.linalg.norm(F)) < 1e-8
            and self._parameters_in_range(u, v)
        ):
            return float(u), float(v)
        return None

    def parameters_at(self, pos):
        p = self.transform.T @ np.asarray(pos, dtype=float)
        if not np.all(np.isfinite(p)):
            return None
        u0 = float(p[0])
        v0 = float(p[1])
        if self._parameters_in_range(u0, v0):
            puv = self._point_local(u0, v0)
            if np.all(np.isfinite(puv)) and float(np.linalg.norm(puv - p)) < 1e-8:
                return u0, v0
        us = np.linspace(self.u_range[0], self.u_range[1], 64)
        vs = np.linspace(self.v_range[0], self.v_range[1], 64)
        U, V = np.meshgrid(us, vs)
        X, Y, Z = self._point_local_array(U, V)
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        dist = np.linalg.norm(pts - p, axis=1)
        finite = np.isfinite(dist)
        if not finite.any():
            return None
        idx = int(np.argmin(np.where(finite, dist, np.inf)))
        u = float(U.ravel()[idx])
        v = float(V.ravel()[idx])
        return self._solve_parameters_point(p, u, v)

    def normal_at(self, pos):
        uv = self.parameters_at(pos)
        if uv is None:
            return None
        return self.normal(uv[0], uv[1])

    def _intersection_seeds(self, o, d, grid_n=64, seed_tol=0.75, max_seeds=12):
        us = np.linspace(self.u_range[0], self.u_range[1], grid_n)
        vs = np.linspace(self.v_range[0], self.v_range[1], grid_n)
        U, V = np.meshgrid(us, vs)
        X, Y, Z = self._point_local_array(U, V)
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        finite = np.isfinite(pts).all(axis=1)
        if not finite.any():
            return []
        Uf = U.ravel()[finite]
        Vf = V.ravel()[finite]
        P = pts[finite]
        rel = P - o
        t = rel @ d
        forward = t > 0.0
        if not forward.any():
            return []
        proj = o + t[:, None] * d
        residual = np.linalg.norm(P - proj, axis=1)
        mask = forward & np.isfinite(residual) & (residual <= seed_tol)
        if not mask.any():
            mask = forward & np.isfinite(residual) & (residual <= 2.0)
            if not mask.any():
                return []
        indices = np.where(mask)[0]
        order = indices[np.argsort(residual[indices])]
        return [
            (float(Uf[idx]), float(Vf[idx]), float(t[idx]))
            for idx in order[:max_seeds]
        ]

    def _accept_intersection(self, o, d, t, u, v, tol):
        p = self._point_local(u, v)
        F = p - (o + t * d)
        if (
            np.all(np.isfinite(F))
            and float(np.linalg.norm(F)) < tol
            and t > 0.0
            and self._parameters_in_range(u, v)
        ):
            return self.transform @ (o + t * d), float(u), float(v)
        return None

    def _newton_intersection(self, o, d, u, v, t):
        for _ in range(50):
            p = self._point_local(u, v)
            F = p - (o + t * d)
            if not np.all(np.isfinite(F)):
                return None
            if float(np.linalg.norm(F)) < 1e-12:
                return self._accept_intersection(o, d, t, u, v, 1e-11)
            ru, rv = self._tangent_local(u, v)
            if not (np.all(np.isfinite(ru)) and np.all(np.isfinite(rv))):
                return None
            A = np.column_stack((-d, ru, rv))
            try:
                delta = np.linalg.solve(A, -F)
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(delta)):
                return None
            scale = max(1.0, abs(t), abs(u), abs(v))
            step_norm = float(np.linalg.norm(delta))
            if step_norm > 0.5 * scale:
                delta = delta * (0.5 * scale / step_norm)
            t += float(delta[0])
            u += float(delta[1])
            v += float(delta[2])
            if not (np.isfinite(t) and np.isfinite(u) and np.isfinite(v)):
                return None
            if step_norm < 1e-14 * scale:
                return self._accept_intersection(o, d, t, u, v, 1e-10)
        return self._accept_intersection(o, d, t, u, v, 1e-10)

    def intersect(self, ray):
        o = self.transform.T @ np.asarray(ray.origin, dtype=float)
        d = self.transform.T @ np.asarray(ray.direction, dtype=float)
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
            return None
        d_norm = float(np.linalg.norm(d))
        if not np.isfinite(d_norm) or d_norm <= 0.0:
            return None
        d = d / d_norm
        for u0, v0, t0 in self._intersection_seeds(o, d):
            hit = self._newton_intersection(o, d, u0, v0, t0)
            if hit is not None:
                return hit
        return None

    def _finite_uv(self, u, v):
        return bool(np.all(np.isfinite(self._point_local(u, v))))

    def _boundary_point(self, u0, v0, u1, v1):
        lo_u, lo_v = float(u0), float(v0)
        hi_u, hi_v = float(u1), float(v1)
        for _ in range(40):
            mid_u = 0.5 * (lo_u + hi_u)
            mid_v = 0.5 * (lo_v + hi_v)
            if mid_u == lo_u and mid_v == lo_v:
                break
            if self._finite_uv(mid_u, mid_v):
                lo_u, lo_v = mid_u, mid_v
            else:
                hi_u, hi_v = mid_u, mid_v
        return self._point_local(lo_u, lo_v)

    def _boundary_faces(self, n, U, V, idmap, a, b, c, d, n_interior):
        counts = (
            (a >= 0).astype(np.int64)
            + (b >= 0).astype(np.int64)
            + (c >= 0).astype(np.int64)
            + (d >= 0).astype(np.int64)
        )
        partial = np.flatnonzero((counts >= 1) & (counts <= 3))
        if partial.size == 0:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
        extra = []
        faces = []
        cache = {}

        def crossing_vertex(i0, j0, i1, j1):
            key = (i0, j0, i1, j1)
            if key not in cache:
                p = self._boundary_point(U[i0, j0], V[i0, j0], U[i1, j1], V[i1, j1])
                if np.all(np.isfinite(p)):
                    cache[key] = n_interior + len(extra)
                    extra.append(p)
                else:
                    cache[key] = None
            return cache[key]

        for q in partial:
            i = int(q // (n - 1))
            j = int(q % (n - 1))
            corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            states = [idmap[ic, jc] >= 0 for ic, jc in corners]
            seq = []
            for k in range(4):
                cur = corners[k]
                nxt = corners[(k + 1) % 4]
                if states[k] != states[(k + 1) % 4]:
                    if states[k]:
                        vi = crossing_vertex(cur[0], cur[1], nxt[0], nxt[1])
                    else:
                        vi = crossing_vertex(nxt[0], nxt[1], cur[0], cur[1])
                    if vi is not None:
                        seq.append(vi)
                if states[(k + 1) % 4]:
                    seq.append(int(idmap[nxt[0], nxt[1]]))
            if len(seq) < 3:
                continue
            for k in range(1, len(seq) - 1):
                faces.append((seq[0], seq[k + 1], seq[k]))
        if not faces:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
        return np.vstack(extra), np.array(faces, dtype=np.int64)

    def mesh(self, n=128):
        n = int(n)
        if n < 2:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
        us = np.linspace(self.u_range[0], self.u_range[1], n)
        vs = np.linspace(self.v_range[0], self.v_range[1], n)
        U, V = np.meshgrid(us, vs)
        X, Y, Z = self._point_local_array(U, V)
        ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
        idmap = np.full((n, n), -1, dtype=np.int64)
        idmap[ok] = np.arange(np.count_nonzero(ok))
        verts = np.empty((np.count_nonzero(ok), 3), dtype=float)
        if np.count_nonzero(ok) > 0:
            verts[:, 0] = X[ok].ravel()
            verts[:, 1] = Y[ok].ravel()
            verts[:, 2] = Z[ok].ravel()
        if idmap.shape[0] < 2 or idmap.shape[1] < 2:
            verts = (self.transform @ verts.T).T
            return verts, np.empty((0, 3), dtype=np.int64)
        i = np.repeat(np.arange(n - 1), n - 1)
        j = np.tile(np.arange(n - 1), n - 1)
        a = idmap[i, j]
        b = idmap[i + 1, j]
        c = idmap[i + 1, j + 1]
        d = idmap[i, j + 1]
        valid = np.minimum.reduce([a, b, c, d]) >= 0
        face_chunks = []
        if valid.any():
            face_chunks.append(np.vstack([
                np.column_stack([a[valid], d[valid], c[valid]]),
                np.column_stack([a[valid], c[valid], b[valid]]),
            ]))
        extra_verts, boundary_faces = self._boundary_faces(n, U, V, idmap, a, b, c, d, len(verts))
        if extra_verts.shape[0] > 0:
            verts = np.vstack([verts, extra_verts]) if verts.shape[0] > 0 else extra_verts
        if boundary_faces.shape[0] > 0:
            face_chunks.append(boundary_faces)
        if not face_chunks:
            verts = (self.transform @ verts.T).T
            return verts, np.empty((0, 3), dtype=np.int64)
        faces = np.vstack(face_chunks)
        cx = float(np.mean(U[ok]))
        cy = float(np.mean(V[ok]))
        with np.errstate(invalid="ignore", divide="ignore"):
            z_center = float(self._z(cx, cy))
        if np.isfinite(z_center):
            z_ref = float(np.median(Z[ok]))
            if z_center < z_ref - 1e-12:
                faces = faces[:, ::-1]
        verts = (self.transform @ verts.T).T
        return verts, faces

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

    @staticmethod
    def _make_surface(entry, transform):
        return Surface(
            entry["x"],
            entry["y"],
            entry["z"],
            transform=transform,
            u_range=entry.get("u_range", (-2.0, 2.0)),
            v_range=entry.get("v_range", (-2.0, 2.0)),
        )

    def refract(self, ray: Ray) -> Ray:
        hit1 = self.surface1.intersect(ray)
        p1, u1, v1 = hit1
        n1 = self.surface1.normal(u1, v1)
        eta1 = 1 / self.n
        transfer1 = eta1 * np.eye(3, 3) + ((np.sqrt(1 - eta1**2 * (1 - np.inner(n1, ray.direction)**2)) / np.inner(n1, ray.direction)) - eta1) * np.outer(n1, n1)
        d1 = transfer1 @ ray.direction
        r1 = Ray(p1, d1)

        hit2 = self.surface2.intersect(r1)
        p2, u2, v2 = hit2
        n2 = self.surface2.normal(u2, v2)
        eta2 = self.n
        transfer2 = eta2 * np.eye(3, 3) + ((np.sqrt(1 - eta2**2 * (1 - np.inner(n2, d1)**2)) / np.inner(n2, d1)) - eta2) * np.outer(n2, n2)
        d2 = transfer2 @ d1

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
