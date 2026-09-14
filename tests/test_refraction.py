"""Refraction physics: Snell's law, TIR, vector form."""

import numpy as np
import pytest

from optics import refract


def test_snell_conserves_n_sin():
    # n1 sin(theta1) = n2 sin(theta2)
    d = np.array([[0.3, 0.0, np.sqrt(1 - 0.09)]])
    n = np.array([[0.0, 0.0, 1.0]])
    dt, cos_i, tir = refract(d, n, 1.0, 1.5)
    sin1 = 0.3
    sin2 = np.sqrt(dt[0, 0] ** 2 + dt[0, 1] ** 2)
    assert not tir[0]
    assert abs(1.0 * sin1 - 1.5 * sin2) < 1e-9
    # Refracted ray bends toward the normal (denser medium): larger z-component.
    assert dt[0, 2] > d[0, 2]
    # Output is a unit vector.
    assert np.allclose(np.linalg.norm(dt[0]), 1.0)


def test_normal_orientation_is_flipped():
    # Normal pointing against the incoming ray must not break the result.
    d = np.array([[0.3, 0.0, np.sqrt(1 - 0.09)]])
    n_up = np.array([[0.0, 0.0, 1.0]])
    n_down = np.array([[0.0, 0.0, -1.0]])
    _, _, _ = refract(d, n_up, 1.0, 1.5)
    dt_up, ci_up, _ = refract(d, n_up, 1.0, 1.5)
    dt_down, ci_down, _ = refract(d, n_down, 1.0, 1.5)
    assert np.allclose(np.abs(ci_up), np.abs(ci_down))
    assert np.allclose(dt_up, dt_down)


def test_normal_incidence_is_unchanged():
    d = np.array([[0.0, 0.0, 1.0]])
    n = np.array([[0.0, 0.0, 1.0]])
    dt, cos_i, tir = refract(d, n, 1.5, 1.0)
    assert np.allclose(dt, d)
    assert cos_i[0] == pytest.approx(1.0)
    assert not tir[0]


def test_tir_dense_to_rare_above_critical():
    # n1=1.5 -> n2=1.0; critical angle arcsin(1/1.5) ~ 41.8 deg; use 60 deg.
    theta = np.radians(60.0)
    d = np.array([[np.sin(theta), 0.0, -np.cos(theta)]])
    n = np.array([[0.0, 0.0, 1.0]])
    _, _, tir = refract(d, n, 1.5, 1.0)
    assert tir[0]


def test_below_critical_does_not_tir():
    theta = np.radians(20.0)
    d = np.array([[np.sin(theta), 0.0, -np.cos(theta)]])
    n = np.array([[0.0, 0.0, 1.0]])
    _, _, tir = refract(d, n, 1.5, 1.0)
    assert not tir[0]


def test_critical_angle_just_above_is_tir():
    # A hair above the critical angle the ray total-internal-reflects.
    theta = np.arcsin(1.0 / 1.5) + np.radians(0.5)
    d = np.array([[np.sin(theta), 0.0, -np.cos(theta)]])
    n = np.array([[0.0, 0.0, 1.0]])
    _, _, tir = refract(d, n, 1.5, 1.0)
    assert tir[0]


def test_batched_refraction():
    N = 5
    thetas = np.linspace(0.0, np.radians(50.0), N)
    d = np.stack([np.sin(thetas), np.zeros(N), np.cos(thetas)], axis=1)
    n = np.tile(np.array([[0.0, 0.0, 1.0]]), (N, 1))
    dt, cos_i, tir = refract(d, n, 1.0, 1.5)
    assert dt.shape == (N, 3)
    assert cos_i.shape == (N,)
    assert tir.shape == (N,)
    sin1 = np.sin(thetas)
    sin2 = np.sqrt(np.sum(dt[:, :2] ** 2, axis=1))
    assert np.allclose(1.0 * sin1, 1.5 * sin2, atol=1e-9)
    assert not np.any(tir)


def test_nonunit_inputs_are_normalized():
    d = np.array([[0.6, 0.0, 2.0]])  # not unit length
    n = np.array([[0.0, 0.0, 1.0]])
    dt, cos_i, tir = refract(d, n, 1.0, 1.5)
    assert np.allclose(np.linalg.norm(dt[0]), 1.0)
    assert not tir[0]
