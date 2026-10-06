import numpy as np


class SurfaceMesher:
    def __init__(self, surface):
        self.surface = surface

    def _valid_uv(self, u, v):
        if not self.surface.accepts_parameters(u, v):
            return False

        return bool(
            np.all(np.isfinite(self.surface.evaluate_local(u, v)))
        )

    def _boundary_point(self, u0, v0, u1, v1):
        lo_u, lo_v = float(u0), float(v0)
        hi_u, hi_v = float(u1), float(v1)
        for _ in range(40):
            mid_u = 0.5 * (lo_u + hi_u)
            mid_v = 0.5 * (lo_v + hi_v)
            if mid_u == lo_u and mid_v == lo_v:
                break
            if self._valid_uv(mid_u, mid_v):
                lo_u, lo_v = mid_u, mid_v
            else:
                hi_u, hi_v = mid_u, mid_v
        return self.surface.evaluate_local(lo_u, lo_v)

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
        surface = self.surface
        n = int(n)
        if n < 2:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
        us = np.linspace(surface.u_range[0], surface.u_range[1], n)
        vs = np.linspace(surface.v_range[0], surface.v_range[1], n)
        U, V = np.meshgrid(us, vs)
        pts = surface.evaluate_local(U, V)
        X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
        ok = surface.aperture_mask(U, V, X, Y, Z)
        idmap = np.full((n, n), -1, dtype=np.int64)
        idmap[ok] = np.arange(np.count_nonzero(ok))
        verts = np.empty((np.count_nonzero(ok), 3), dtype=float)
        if np.count_nonzero(ok) > 0:
            verts[:, 0] = X[ok].ravel()
            verts[:, 1] = Y[ok].ravel()
            verts[:, 2] = Z[ok].ravel()
        if idmap.shape[0] < 2 or idmap.shape[1] < 2:
            verts = surface.transform.points_to_world(verts)
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
            verts = surface.transform.points_to_world(verts)
            return verts, np.empty((0, 3), dtype=np.int64)
        faces = np.vstack(face_chunks)
        cx = float(np.mean(U[ok]))
        cy = float(np.mean(V[ok]))
        with np.errstate(invalid="ignore", divide="ignore"):
            z_center = float(surface._z(cx, cy))
        if np.isfinite(z_center):
            z_ref = float(np.median(Z[ok]))
            if z_center < z_ref - 1e-12:
                faces = faces[:, ::-1]
        verts = surface.transform.points_to_world(verts)
        return verts, faces
