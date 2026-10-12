import numpy as np

import optics.surface as surface_mod


def _sphere_surface(make_element):
    return make_element("sqrt(1 - (x**2) - (y**2))", "0").surface1


def test_njit_path_active_for_standard_surface(make_element):
    s = _sphere_surface(make_element)
    assert s._eval_njit is not None
    assert s._deriv_njit is not None
    desc = s.njit_descriptor
    assert desc is not None
    assert desc[0] is s._eval_cfn
    assert desc[1] is s._deriv_cfn


def test_scalar_eval_matches_array_lambda(make_element):
    s = _sphere_surface(make_element)
    rng = np.random.default_rng(0)
    u = rng.uniform(-1.5, 1.5, 64)
    v = rng.uniform(-1.5, 1.5, 64)
    arr = s.evaluate_local(u, v)
    for i in range(u.size):
        scalar = s.evaluate_local(float(u[i]), float(v[i]))
        np.testing.assert_allclose(
            scalar, arr[i], rtol=1e-12, atol=1e-12, equal_nan=True
        )


def test_out_of_domain_scalar_is_nan(make_element):
    s = _sphere_surface(make_element)
    p = s.evaluate_local(1.5, 0.0)
    assert np.isnan(p[2])


def test_division_by_zero_matches_lambda(make_element):
    s = make_element("u / (v - 0.5)", "0").surface1
    u = np.array([0.3, 0.0, 0.3])
    v = np.array([0.5, 0.5, 0.25])
    arr = s.evaluate_local(u, v)
    for i in range(u.size):
        scalar = s.evaluate_local(float(u[i]), float(v[i]))
        np.testing.assert_allclose(
            scalar, arr[i], rtol=1e-12, atol=1e-12, equal_nan=True
        )
    assert np.isinf(arr[0, 2])
    assert np.isnan(arr[1, 2])


def test_derivatives_match_finite_differences(make_element):
    s = make_element("x**2 + 3*y + x*y**3", "0").surface1
    h = 1e-5
    for u, v in [(0.3, -0.7), (1.2, 0.4), (-0.9, 1.1)]:
        ru, rv = s.derivatives_local(u, v)
        fd_u = (s.evaluate_local(u + h, v) - s.evaluate_local(u - h, v)) / (2 * h)
        fd_v = (s.evaluate_local(u, v + h) - s.evaluate_local(u, v - h)) / (2 * h)
        np.testing.assert_allclose(ru, fd_u, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(rv, fd_v, rtol=1e-5, atol=1e-5)


def test_codegen_fallback_keeps_lambda_path(make_element, monkeypatch):
    monkeypatch.setattr(
        surface_mod,
        "_build_njit_functions",
        lambda *args, **kwargs: (None, None, None, None),
    )
    cache_clear = getattr(surface_mod._compile_parametric_surface, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    try:
        s = _sphere_surface(make_element)
        assert s._eval_njit is None
        assert s.njit_descriptor is None
        np.testing.assert_allclose(
            s.point(0.3, 0.2), [0.3, 0.2, np.sqrt(1.0 - 0.13)], rtol=1e-12
        )
    finally:
        if cache_clear is not None:
            cache_clear()
