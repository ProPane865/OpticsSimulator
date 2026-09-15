"""End-to-end ray tracing through optical stacks."""

import numpy as np
import pytest

from optics import (ConicSurface, Plane, PolySagSurface, OpticalStack,
                    IrradianceSource, accel)


@pytest.fixture(autouse=True)
def _numba_required():
    assert accel.HAVE_NUMBA is True, "Numba is a required backend"


def _lens_stack(R=50.0, n=1.5, z1=5.0, aperture=None):
    stack = OpticalStack("lens")
    stack.add_surface(ConicSurface("l1", radius=R, conicity=0.0, z0=0.0,
                                   aperture=aperture), medium_index=n)
    stack.add_surface(ConicSurface("l2", radius=-R, conicity=0.0, z0=z1,
                                   aperture=aperture), medium_index=n)
    stack.add_media(1.0)
    return stack


def _gaussian_source(grid=80, pitch=1.0, origin=(0, 0, -20.0), sigma_frac=0.2):
    pat = np.exp(-((np.arange(grid) - grid / 2) ** 2) / (2 * (grid * sigma_frac) ** 2))
    pat = (pat[None, :] * pat[:, None]).astype(np.float32)
    return IrradianceSource(pat, pitch=pitch, origin=origin, direction=(0, 0, 1))


def test_trace_output_shapes():
    stack = _lens_stack()
    src = _gaussian_source(grid=64)
    tr = stack.trace(src)
    S, N, _ = tr.hits.shape
    assert S == 2
    assert N == 64 * 64
    assert tr.tir.shape == (S, N)
    assert tr.blocked.shape == (N,)
    assert tr.irradiance_input.shape == (N,)
    assert tr.uv.shape == (N, 2)
    assert tr.grid_shape == (64, 64)


def test_trace_is_numba_backend():
    stack = _lens_stack()
    tr = stack.trace(_gaussian_source(grid=32))
    assert accel.HAVE_NUMBA is True
    assert tr.surfaces[0].kind == 1  # CONIC


def test_collimated_lens_focuses():
    # A converging bi-convex lens should focus a collimated beam: spot shrinks.
    stack = _lens_stack(R=50.0)
    src = _gaussian_source(grid=80, sigma_frac=0.2)
    tr = stack.trace(src)
    fz = 50.0 / (1.5 - 1.0)  # paraxial focal length of thin lens ~ R/(2(n-1))
    pts_exit = tr.hits[-1]
    spot_exit = np.sqrt(np.mean(pts_exit[:, 0] ** 2 + pts_exit[:, 1] ** 2))
    pts_focus, _ = tr.propagate_to(fz)
    spot_focus = np.sqrt(np.mean(pts_focus[:, 0] ** 2 + pts_focus[:, 1] ** 2))
    assert spot_focus < spot_exit


def test_focus_plane_is_near_paraxial_focal_length():
    stack = _lens_stack(R=50.0)
    tr = stack.trace(_gaussian_source(grid=80, sigma_frac=0.3))
    pts0 = tr.hits[-1]
    zs = np.linspace(30, 90, 61)
    spots = []
    for z in zs:
        p, _ = tr.propagate_to(z)
        spots.append(np.sqrt(np.mean(p[:, 0] ** 2 + p[:, 1] ** 2)))
    spots = np.array(spots)
    fz = float(zs[np.argmin(spots)])
    # Bi-convex lens (R=50, n=1.5) focuses around z ~ 65-70; well past the lens.
    assert 45.0 < fz < 85.0


def test_aperture_blocks_outer_rays():
    stack = _lens_stack(aperture=40.0)
    src = _gaussian_source(grid=80)
    tr = stack.trace(src)
    n_blocked = int(tr.blocked.sum())
    assert n_blocked > 0
    # Blocked rays must be a strict subset of total.
    assert n_blocked < tr.n_rays


def test_no_aperture_blocks_nothing():
    stack = _lens_stack(aperture=None)
    src = _gaussian_source(grid=80)
    tr = stack.trace(src)
    assert int(tr.blocked.sum()) == 0


def test_aperture_radius_matches():
    # With aperture=40, hit points on the first surface must satisfy r <= 40.
    stack = _lens_stack(aperture=40.0)
    src = _gaussian_source(grid=64)
    tr = stack.trace(src)
    hits = tr.hits[0]
    blocked = tr.blocked
    for i in range(hits.shape[0]):
        if not blocked[i]:
            r = np.sqrt(hits[i, 0] ** 2 + hits[i, 1] ** 2)
            assert r <= 40.0 + 1e-6


def test_ray_origin_recorded():
    stack = _lens_stack()
    src = _gaussian_source(grid=16, origin=(0, 0, -20.0))
    tr = stack.trace(src)
    assert tr.source_points.shape == (16 * 16, 3)
    assert np.allclose(tr.source_points[:, 2], -20.0)


def test_plane_surface_traces_straight():
    stack = OpticalStack("plane")
    stack.add_surface(Plane("p", z0=10.0), medium_index=1.5)
    stack.add_media(1.0)
    pat = np.ones((4, 4), dtype=np.float32)
    src = IrradianceSource(pat, pitch=1.0, origin=(0, 0, 0.0), direction=(0, 0, 1))
    tr = stack.trace(src)
    # Rays travel straight through a plane; z at first surface == 10.
    assert np.allclose(tr.hits[0][:, 2], 10.0)


def test_plane_aperture_blocks_outer_rays():
    stack = OpticalStack("plane")
    stack.add_surface(Plane("p", z0=10.0, aperture=3.0), medium_index=1.5)
    stack.add_media(1.0)
    tr = stack.trace(_gaussian_source(grid=80))
    hits = tr.hits[0]
    blocked = tr.blocked
    n_blocked = int(blocked.sum())
    assert n_blocked > 0
    assert n_blocked < tr.n_rays
    for i in range(hits.shape[0]):
        if not blocked[i]:
            r = np.sqrt(hits[i, 0] ** 2 + hits[i, 1] ** 2)
            assert r <= 3.0 + 1e-6


def test_poly_surface_traces():
    P = PolySagSurface("poly", order=2)
    P.coeffs[2, 0] = 0.001  # weak focusing term
    stack = OpticalStack("poly")
    stack.add_surface(P, medium_index=1.5)
    stack.add_media(1.0)
    tr = stack.trace(_gaussian_source(grid=32))
    assert tr.hits.shape[0] == 1
    assert not np.any(np.isnan(tr.hits[0]))


def test_tir_flag_can_be_set():
    # A 60-degree incidence on a 1.5->1.0 boundary at the second (concave) surface
    # of a steep lens can produce TIR for marginal rays. Just check the flag dtype.
    stack = _lens_stack()
    tr = stack.trace(_gaussian_source(grid=32))
    assert tr.tir.dtype == bool
    # At least the central on-axis rays do not TIR.
    center = tr.tir[:, (32 * 32) // 2]
    assert not np.any(center)
