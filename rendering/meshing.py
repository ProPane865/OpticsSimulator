from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import numpy as np

if TYPE_CHECKING:
    from optics.primitive import Surface

_MESH_CACHE: "WeakKeyDictionary[object, dict[int, tuple[np.ndarray, np.ndarray]]]" = WeakKeyDictionary()


class SurfaceMesher:
    def __init__(self, surface: "Surface") -> None:
        self.surface = surface

    def _valid_uv_array(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
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

    def _bisect_edges(self, edge_lo: list, edge_hi: list) -> np.ndarray:
        surface = self.surface
        lo = np.array(edge_lo, dtype=float)
        hi = np.array(edge_hi, dtype=float)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            ok = self._valid_uv_array(mid[:, 0], mid[:, 1])[:, None]
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        return surface.evaluate_local(lo[:, 0], lo[:, 1])

    def _boundary_faces(
        self,
        n: int,
        U: np.ndarray,
        V: np.ndarray,
        idmap: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        n_interior: int,
    ) -> "tuple[np.ndarray, np.ndarray]":
        counts = (
            (a >= 0).astype(np.int64)
            + (b >= 0).astype(np.int64)
            + (c >= 0).astype(np.int64)
            + (d >= 0).astype(np.int64)
        )
        partial = np.flatnonzero((counts >= 1) & (counts <= 3))
        if partial.size == 0:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
        edge_index: dict[tuple[int, int, int, int], int] = {}
        edge_lo: list = []
        edge_hi: list = []
        faces = []

        def crossing_vertex(i0: int, j0: int, i1: int, j1: int) -> int:
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

    def mesh(self, n: int = 128) -> "tuple[np.ndarray, np.ndarray]":
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

    def _compute_mesh(self, n: int) -> "tuple[np.ndarray, np.ndarray]":
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
        valid = (np.minimum(np.minimum(a, b), np.minimum(c, d)) >= 0)
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
        # Global winding fix. Faces are emitted with a fixed winding, so a
        # single flip makes the whole patch consistent. We orient each surface
        # so its geometric normals point "outward" in the rendering sense: away
        # from the flat side of the patch (the side the surface bulges toward).
        # This is intentionally the OPPOSITE of Surface.normal (which is
        # -cross(ru, rv) and points -z for a graph-over-XY surface); refraction
        # re-orients normals per-ray via _orient, so the two conventions are
        # independent. The heuristic compares the surface height at the patch
        # centre to the median height: a convex bulge flips the winding.
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
