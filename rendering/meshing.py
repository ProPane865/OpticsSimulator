import numpy as np
from weakref import WeakKeyDictionary


_MESH_CACHE = WeakKeyDictionary()


class SurfaceMesher:
    def __init__(self, surface):
        self.surface = surface

    def _valid_uv(self, u, v):
        if not self.surface.accepts_parameters(u, v):
            return False

        return bool(
            np.all(np.isfinite(self.surface.evaluate_local(u, v)))
        )

    def _valid_uv_array(self, u, v):
        surface = self.surface
        ok = surface.in_domain(u, v)
        pts = surface.evaluate_local(u, v)
        ok &= np.all(np.isfinite(pts), axis=-1)
        if surface.aperture is not None:
            ok &= np.asarray(
                surface.aperture.mask(
                    surface, u, v, pts[..., 0], pts[..., 1], pts[..., 2]
                ),
                dtype=bool,
            )
        return ok

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

    def _bisect_edges(self, edge_lo, edge_hi):
        surface = self.surface
        lo = np.array(edge_lo, dtype=float)
        hi = np.array(edge_hi, dtype=float)
        try:
            self._valid_uv_array(lo[:, 0], lo[:, 1])
        except NotImplementedError:
            return np.array(
                [
                    self._boundary_point(lu, lv, hu, hv)
                    for (lu, lv), (hu, hv) in zip(edge_lo, edge_hi)
                ],
                dtype=float,
            )
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            ok = self._valid_uv_array(mid[:, 0], mid[:, 1])[:, None]
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        return surface.evaluate_local(lo[:, 0], lo[:, 1])

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
        edge_index = {}
        edge_lo = []
        edge_hi = []
        faces = []

        def crossing_vertex(i0, j0, i1, j1):
            key = (i0, j0, i1, j1)
            idx = edge_index.get(key)
            if idx is None:
                idx = len(edge_lo)
                edge_index[key] = idx
                edge_lo.append((U[i0, j0], V[i0, j0]))
                edge_hi.append((U[i1, j1], V[i1, j1]))
            return n_interior + idx

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
                    seq.append(vi)
                if states[(k + 1) % 4]:
                    seq.append(int(idmap[nxt[0], nxt[1]]))
            if len(seq) < 3:
                continue
            for k in range(1, len(seq) - 1):
                faces.append((seq[0], seq[k + 1], seq[k]))
        if not faces:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
        extra = self._bisect_edges(edge_lo, edge_hi)
        finite = np.all(np.isfinite(extra), axis=1)
        if not finite.all():
            remap = np.full(len(extra), -1, dtype=np.int64)
            remap[finite] = np.arange(int(np.count_nonzero(finite)))
            clean = []
            for f in faces:
                g = []
                ok = True
                for idx in f:
                    if idx >= n_interior:
                        r = int(remap[idx - n_interior])
                        if r < 0:
                            ok = False
                            break
                        g.append(n_interior + r)
                    else:
                        g.append(idx)
                if ok:
                    clean.append(tuple(g))
            if not clean:
                return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
            return extra[finite], np.array(clean, dtype=np.int64)
        return extra, np.array(faces, dtype=np.int64)

    def mesh(self, n=128):
        n = int(n)
        per_surface = _MESH_CACHE.get(self.surface)
        if per_surface is None:
            per_surface = {}
            _MESH_CACHE[self.surface] = per_surface
        cached = per_surface.get(n)
        if cached is not None:
            return cached
        result = self._compute_mesh(n)
        per_surface[n] = result
        return result

    def _compute_mesh(self, n):
        surface = self.surface
        n = int(n)
        if n < 2:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
        (u_min, u_max), (v_min, v_max) = surface.parameter_range
        us = np.linspace(u_min, u_max, n)
        vs = np.linspace(v_min, v_max, n)
        U, V = np.meshgrid(us, vs)
        pts = surface.evaluate_local(U, V)
        X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
        ok = surface.aperture_mask(U, V, X, Y, Z)
        idmap = np.full((n, n), -1, dtype=np.int64)
        idmap[ok] = np.arange(np.count_nonzero(ok))
        verts = pts[ok]
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
            center = surface.evaluate_local(cx, cy)
            if np.all(np.isfinite(center)):
                z_center = center[2]
                z_ref = float(np.median(Z[ok]))

                if z_center < z_ref - 1e-12:
                    faces = faces[:, ::-1]
        verts = surface.transform.points_to_world(verts)
        return verts, faces
