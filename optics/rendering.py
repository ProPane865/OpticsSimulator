"""Geometry helpers for GUI rendering: surface meshes, source quads, ray lines."""

from __future__ import annotations

import numpy as np


def surface_mesh(surface, rmax, n=64):
    """Triangular mesh of a sag surface over ``[-rmax, rmax]`` squared,
    clipped to the circular aperture of radius ``rmax``.

    Faces with any vertex outside the circle are discarded, so sag values
    outside the surface's valid domain (e.g. conic corners beyond the
    radius of curvature) are never rendered.

    Returns ``(vertices (M, 3), faces (K, 3))``.
    """
    xs = np.linspace(-rmax, rmax, n)
    X, Y = np.meshgrid(xs, xs)
    Z = np.asarray(surface.sag(X, Y), dtype=float)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    i = np.arange(n - 1)[:, None]
    j = np.arange(n - 1)[None, :]
    a = i * n + j
    b = a + 1
    c = a + n
    d = c + 1
    faces = np.concatenate([
        np.stack([a, b, d], axis=-1).reshape(-1, 3),
        np.stack([a, d, c], axis=-1).reshape(-1, 3),
    ])

    # Clip mesh to the circular aperture.
    r2 = verts[:, 0] ** 2 + verts[:, 1] ** 2
    inside = r2 <= rmax ** 2
    faces = faces[np.all(inside[faces], axis=1)]

    return verts.astype(np.float32), faces.astype(np.int32)


def source_quad(source):
    """Corners, UVs and pattern for a textured source plane.

    Returns ``(corners (4, 3), uvs (4, 2), pattern (H, W))``.
    """
    H, W = source.shape
    pu, pv = source.pitch
    ox, oy, oz = source.origin
    corners = np.array([
        [ox - W * pu / 2, oy - H * pv / 2, oz],
        [ox + W * pu / 2, oy - H * pv / 2, oz],
        [ox + W * pu / 2, oy + H * pv / 2, oz],
        [ox - W * pu / 2, oy + H * pv / 2, oz],
    ], dtype=np.float32)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return corners, uvs, source.pattern.astype(np.float32)


def ray_polylines(trace, detector_z=None):
    """Ray paths as line segments from source through surfaces to the detector.

    Returns ``(segments, detector_hits)`` where ``segments`` has shape
    ``(n_rays * (n_surfaces + 1), 2, 3)`` — one ``(start, end)`` pair per
    segment, grouped per ray — and ``detector_hits`` has shape ``(n_rays, 3)``.
    """
    base = [trace.source_points]
    for s in range(trace.n_surfaces):
        base.append(trace.hits[s])
    if detector_z is None:
        detector_z = float(trace.hits[-1][:, 2].max()) + 1.0
    det_pts, _t = trace.propagate_to(detector_z)
    base.append(det_pts)

    path = np.stack(base, axis=1)          # (N, n_seg+1, 3)
    segments = np.stack([path[:, :-1], path[:, 1:]], axis=2)  # (N, segs, 2, 3)
    return segments.reshape(-1, 2, 3).astype(np.float32), det_pts
