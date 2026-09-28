import numpy as np


def test_plane_equation_constant(make_element):
    elem = make_element("2", "-2")
    f = elem.surface1[0]
    assert np.isclose(f(0.0, 0.0), 2.0)
    assert np.isclose(f(0.5, -0.25), 2.0)


def test_polynomial_surface_values(make_element):
    elem = make_element("1 + x**2 + 4*y", "0")
    f = elem.surface1[0]
    assert np.isclose(f(0.5, 1.0), 5.25)
    assert np.isclose(f(0.0, 0.0), 1.0)


def test_sphere_surface_value(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    f = elem.surface1[0]
    assert np.isclose(f(0.5, 0.0), np.sqrt(0.75))
    assert np.isclose(f(0.0, 0.0), 1.0)


def test_first_derivatives(make_element):
    elem = make_element("x**2 + 3*y", "0")
    _, fx, fy = elem.surface1
    assert np.isclose(fx(1.0, 2.0), 2.0)
    assert np.isclose(fy(1.0, 2.0), 3.0)


def test_out_of_domain_returns_nan(make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    f = elem.surface1[0]
    assert np.isnan(f(1.5, 0.0))
