"""Rendering geometry tests: surface mesh aperture clipping, ray segment layout."""

import numpy as np
import pytest

from optics import ConicSurface, Plane
from optics.rendering import ray_polylines, surface_mesh
from optics.stack import TraceResult


def _mock_trace(n_rays, n_surfaces, seed=0):
    rng = np.random.default_rng(seed)
    hits = []
    for s in range(n_surfaces):
        z = 10.0 * (s + 1)
        hits.append(np.column_stack([
            rng.normal(0.0, 2.0, n_rays),
            rng.normal(0.0, 2.0, n_rays),
            np.full(n_rays, z),
        ]))
    tr = TraceResult(
        surfaces=list(range(n_surfaces)),
        hits=np.stack(hits, axis=0),
        tir=np.zeros((n_surfaces, n_rays)),
        final_dirs=np.tile([0.1, -0.05, 1.0], (n_rays, 1)),
    )
    tr.source_points = np.column_stack([
        rng.normal(0.0, 3.0, n_rays),
        rng.normal(0.0, 3.0, n_rays),
        np.zeros(n_rays),
    ])
    return tr


@pytest.mark.parametrize(("n_rays", "n_surfaces"), [(2, 1), (4, 3), (5, 2)])
def test_ray_polylines_segments_are_source_to_detector(n_rays, n_surfaces):
    tr = _mock_trace(n_rays, n_surfaces)
    n_seg = n_surfaces + 1
    segs, det_pts = ray_polylines(tr, detector_z=50.0)

    assert segs.shape == (n_rays * n_seg, 2, 3)
    assert segs.dtype == np.float32
    assert det_pts.shape == (n_rays, 3)

    # Same reshape the viewport applies in _build_rays.
    per_ray = segs.reshape(n_rays, n_seg, 2, 3)

    # Each segment's end must equal the next segment's start: endpoints
    # belong to the same segment, not to unrelated ones.
    np.testing.assert_allclose(
        per_ray[:, 1:, 0], per_ray[:, :-1, 1], rtol=0.0, atol=1e-5)
    # The polyline starts at the ray origin and ends at the detector point.
    np.testing.assert_allclose(
        per_ray[:, 0, 0], tr.source_points, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(
        per_ray[:, -1, 1], det_pts, rtol=0.0, atol=1e-5)


@pytest.fixture(scope="module")
def _conic_mesh():
    # R = 50 mm sphere, circular aperture 45 mm: the unclipped square grid
    # reaches corners at r = 45*sqrt(2) ~ 63.64 mm, outside the spherical
    # domain (r <= 50) where sag() clamps to an artificial flared sheet.
    surf = ConicSurface("lens", radius=50.0, conicity=0.0, z0=0.0, aperture=45.0)
    return surface_mesh(surf, 45.0, n=48)


def test_surface_mesh_clipped_to_circular_aperture(_conic_mesh):
    verts, faces = _conic_mesh
    n = 48

    assert faces.shape[0] > 0
    assert faces.max() < len(verts)
    # The square grid has 2*(n-1)^2 faces; clipping must discard some.
    assert faces.shape[0] < 2 * (n - 1) ** 2

    r = np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2)
    r_face = r[faces]
    # Every face vertex lies inside the circular aperture (float32 eps).
    assert np.all(r_face <= 45.0 + 1e-4)
    # And nowhere near the old square corner radius 45*sqrt(2) ~ 63.64.
    assert r_face.max() < 60.0


def test_surface_mesh_conic_sag_stays_in_domain(_conic_mesh):
    verts, faces = _conic_mesh
    # With sag clamping removed by clipping, edge sag at r=45 is
    # c*r^2/(1+sqrt(1-c^2 r^2)) ~ 28.2 mm, not the ~81 mm clamped corner.
    z_face = verts[faces, 2]
    assert z_face.max() < 40.0


def test_surface_mesh_plane_clipped_to_circle():
    surf = Plane("plane", z0=2.0)
    verts, faces = surface_mesh(surf, 30.0, n=32)

    assert faces.shape[0] > 0
    assert faces.max() < len(verts)
    assert faces.shape[0] < 2 * 31 ** 2

    r = np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2)
    assert np.all(r[faces] <= 30.0 + 1e-4)
    np.testing.assert_allclose(verts[:, 2], 2.0, atol=1e-5)
