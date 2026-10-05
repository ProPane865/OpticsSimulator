import numpy as np
import pytest

from optics.refraction import Ray


def test_direction_normalized_to_unit():
    ray = Ray(np.array([0.0, 0.0, 0.0]), np.array([3.0, 4.0, 0.0]))
    np.testing.assert_allclose(ray.direction, [0.6, 0.8, 0.0], atol=1e-12)


def test_origin_preserved_verbatim():
    origin = np.array([1.0, -2.0, 3.0])
    ray = Ray(origin, np.array([0.0, 0.0, -5.0]))
    np.testing.assert_array_equal(ray.origin, origin)
    np.testing.assert_allclose(ray.direction, [0.0, 0.0, -1.0], atol=1e-12)


def test_scalar_multiple_direction_normalized():
    ray = Ray(np.zeros(3), 7.0 * np.array([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(ray.direction, [0.0, 0.0, -1.0], atol=1e-12)

def test_zero_direction_rejected():
    with pytest.raises(ValueError):
        Ray(np.zeros(3), np.zeros(3))