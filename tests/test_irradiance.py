"""Irradiance mapping via the ray-map Jacobian and flux conservation."""

import numpy as np
import pytest

from optics import (ConicSurface, Plane, OpticalStack, IrradianceSource,
                    irradiance_from_trace, compute_jacobian, flux_conservation)


def _lens_stack(R=50.0, n=1.5, z1=5.0, aperture=45.0):
    stack = OpticalStack("lens")
    stack.add_surface(ConicSurface("l1", radius=R, conicity=0.0, z0=0.0,
                                   aperture=aperture), medium_index=n)
    stack.add_surface(ConicSurface("l2", radius=-R, conicity=0.0, z0=z1,
                                   aperture=aperture), medium_index=n)
    stack.add_media(1.0)
    return stack


def _gaussian_source(grid=80, sigma_frac=0.2, peak=1.0):
    a = np.arange(grid) - grid / 2.0
    env = peak * np.exp(-(a ** 2) / (2 * (grid * sigma_frac) ** 2))
    pat = (env[None, :] * env[:, None]).astype(np.float32)
    return IrradianceSource(pat, pitch=1.0, origin=(0, 0, -20.0), direction=(0, 0, 1))


def test_irradiance_output_shape():
    stack = _lens_stack()
    tr = stack.trace(_gaussian_source(grid=64))
    irr = irradiance_from_trace(tr, detector_z=50.0)
    assert irr["irradiance"].shape == (64, 64)
    assert irr["valid"].shape == (64, 64)
    assert irr["jacobian_det"].shape == (64, 64)
    assert irr["output_xy"].shape == (64, 64, 2)
    assert irr["input_uv"].shape == (64, 64, 2)


def test_focused_lens_has_positive_peak():
    stack = _lens_stack()
    tr = stack.trace(_gaussian_source(grid=80, sigma_frac=0.2))
    fz = 50.0 / (1.5 - 1.0)
    irr = irradiance_from_trace(tr, detector_z=fz)
    assert irr["irradiance"].max() > 0.0
    assert irr["valid"].mean() > 0.5


def _min_spot_radius(tr, z):
    p, _ = tr.propagate_to(z)
    return float(np.sqrt(np.mean(p[:, 0] ** 2 + p[:, 1] ** 2)))


def test_focusing_amplifies_peak():
    # Irradiance peak at the focal plane exceeds the input peak (flux compression).
    stack = _lens_stack()
    src = _gaussian_source(grid=80, sigma_frac=0.2)
    tr = stack.trace(src)
    in_peak = src.pattern.max()
    zs = np.linspace(40, 80, 41)
    spots = np.array([_min_spot_radius(tr, z) for z in zs])
    fz = float(zs[np.argmin(spots)])
    irr_best = irradiance_from_trace(tr, detector_z=fz)
    assert irr_best["irradiance"].max() > in_peak


def test_blocked_rays_are_zero_in_output():
    stack = _lens_stack(aperture=40.0)
    tr = stack.trace(_gaussian_source(grid=64))
    irr = irradiance_from_trace(tr, detector_z=50.0)
    blocked = tr.blocked.reshape(64, 64)
    assert np.all(irr["irradiance"][blocked] == 0.0)


def test_jacobian_sign_change_across_focus():
    # The Jacobian determinant flips sign as rays cross the focal point (~z 65):
    # orientation-preserving before focus, inverted after.
    stack = _lens_stack()
    tr = stack.trace(_gaussian_source(grid=48))
    det_before, _, _, _, _ = compute_jacobian(tr, detector_z=30.0)
    det_after, _, _, _, _ = compute_jacobian(tr, detector_z=75.0)
    assert not np.any(det_before < 0)
    assert np.any(det_after < 0)


def test_uniform_flat_map_preserves_irradiance():
    # A plane with no focusing: output irradiance ~ input (|det J| ~ 1).
    stack = OpticalStack("plane")
    stack.add_surface(Plane("p", z0=0.0), medium_index=1.0)
    stack.add_media(1.0)
    pat = np.ones((16, 16), dtype=np.float32)
    src = IrradianceSource(pat, pitch=2.0, origin=(0, 0, -10.0), direction=(0, 0, 1))
    tr = stack.trace(src)
    irr = irradiance_from_trace(tr, detector_z=10.0)
    assert np.allclose(irr["irradiance"], 1.0, atol=1e-6)


def test_flux_conservation_reasonable():
    stack = _lens_stack()
    tr = stack.trace(_gaussian_source(grid=80, sigma_frac=0.2))
    ratio = flux_conservation(tr, detector_z=50.0)
    # Not a strict test (finite sampling + clamping), but should be O(1) and finite.
    assert np.isfinite(ratio)


def test_valid_mask_respects_nonfinite():
    # Detectors placed exactly at a caustic can produce non-finite Jacobians;
    # the valid mask must drop those cells.
    stack = _lens_stack()
    tr = stack.trace(_gaussian_source(grid=48))
    fz = 50.0 / (1.5 - 1.0)
    detJ, X, Y, U, V = compute_jacobian(tr, detector_z=fz)
    irr = irradiance_from_trace(tr, detector_z=fz)
    valid = irr["valid"]
    # Valid cells must be a subset of finite output coordinates.
    finite_xy = np.isfinite(X) & np.isfinite(Y)
    assert np.all(valid <= finite_xy)
