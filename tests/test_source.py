import numpy as np
import pytest

from optics.primitive import Ray
from optics.source import (
    POINT_ARRAY,
    SINGLE_POINT,
    PointSource,
    fibonacci_cone_directions,
)


def test_default_source_attributes():
    source = PointSource([0.0, 0.0, 3.0])
    assert np.allclose(source.position, [0.0, 0.0, 3.0])
    assert source.power == pytest.approx(1.0)
    assert source.num_rays == 64
    assert source.source_type == SINGLE_POINT


def test_invalid_position_raises():
    with pytest.raises(ValueError):
        PointSource([1.0, 2.0])
    with pytest.raises(ValueError):
        PointSource([np.nan, 0.0, 0.0])


def test_invalid_power_raises():
    with pytest.raises(ValueError):
        PointSource([0.0, 0.0, 0.0], power=-1.0)
    with pytest.raises(ValueError):
        PointSource([0.0, 0.0, 0.0], power=np.nan)


def test_invalid_num_rays_raises():
    with pytest.raises(ValueError):
        PointSource([0.0, 0.0, 0.0], num_rays=0)


def test_emit_rays_count_and_type():
    source = PointSource([1.0, -2.0, 3.0], num_rays=37)
    rays = source.emit_rays()
    assert len(rays) == 37
    assert all(isinstance(r, Ray) for r in rays)


def test_emit_rays_origins_at_source():
    position = [1.0, -2.0, 3.0]
    source = PointSource(position, num_rays=16)
    for ray in source.emit_rays():
        assert np.allclose(ray.origin, position)


def test_emit_rays_directions_are_unit_and_spherical():
    source = PointSource([0.0, 0.0, 0.0], num_rays=128)
    directions = np.array([ray.direction for ray in source.emit_rays()])
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.linalg.norm(directions.mean(axis=0)) < 0.1
    for axis in range(3):
        assert directions[:, axis].min() < -0.5
        assert directions[:, axis].max() > 0.5


def test_emit_rays_deterministic():
    source = PointSource([0.0, 0.0, 0.0], num_rays=24)
    first = [ray.direction for ray in source.emit_rays()]
    second = [ray.direction for ray in source.emit_rays()]
    for a, b in zip(first, second):
        assert np.allclose(a, b)


def test_fibonacci_full_cone_is_sphere():
    assert fibonacci_cone_directions(0, 360.0).shape == (0, 3)
    dirs = fibonacci_cone_directions(10, 360.0)
    assert dirs.shape == (10, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)


def test_point_array_not_implemented():
    source = PointSource([0.0, 0.0, 0.0], source_type=POINT_ARRAY)
    with pytest.raises(NotImplementedError):
        source.emit_rays()


def test_unknown_source_type_raises():
    source = PointSource([0.0, 0.0, 0.0], source_type="laser")
    with pytest.raises(ValueError):
        source.emit_rays()


def test_default_cone_angle_is_full_sphere():
    source = PointSource([0.0, 0.0, 0.0])
    assert source.cone_angle == pytest.approx(360.0)


def test_invalid_cone_angle_raises():
    with pytest.raises(ValueError):
        PointSource([0.0, 0.0, 0.0], cone_angle=-1.0)
    with pytest.raises(ValueError):
        PointSource([0.0, 0.0, 0.0], cone_angle=361.0)
    with pytest.raises(ValueError):
        PointSource([0.0, 0.0, 0.0], cone_angle=np.nan)


def test_cone_angle_preserves_ray_count():
    for angle in (0.0, 30.0, 90.0, 180.0, 360.0):
        source = PointSource([0.0, 0.0, 0.0], num_rays=48, cone_angle=angle)
        assert len(source.emit_rays()) == 48


def test_cone_directions_within_cone_and_unit():
    dirs = fibonacci_cone_directions(128, 90.0)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    cos_half = np.cos(np.deg2rad(45.0))
    assert np.all(dirs[:, 2] <= -cos_half + 1e-12)


def test_cone_directions_concentrated_downwards():
    wide = fibonacci_cone_directions(256, 360.0)
    narrow = fibonacci_cone_directions(256, 30.0)
    mean_narrow = narrow.mean(axis=0)
    assert abs(mean_narrow[0]) < 0.01
    assert abs(mean_narrow[1]) < 0.01
    assert mean_narrow[2] < -0.9
    assert np.linalg.norm(wide.mean(axis=0)) < 0.1
    assert np.linalg.norm(mean_narrow) > np.linalg.norm(wide.mean(axis=0))


def test_cone_zero_all_rays_point_down():
    dirs = fibonacci_cone_directions(16, 0.0)
    assert np.allclose(dirs, np.array([0.0, 0.0, -1.0]))


def test_cone_invalid_angle_raises():
    with pytest.raises(ValueError):
        fibonacci_cone_directions(8, 400.0)
    with pytest.raises(ValueError):
        fibonacci_cone_directions(8, -10.0)


def test_emit_rays_uses_cone_angle():
    source = PointSource([0.0, 0.0, 3.0], num_rays=64, cone_angle=90.0)
    directions = np.array([ray.direction for ray in source.emit_rays()])
    cos_half = np.cos(np.deg2rad(45.0))
    assert np.all(directions[:, 2] <= -cos_half + 1e-12)


def test_emit_rays_power_divided_by_ray_count():
    source = PointSource([0.0, 0.0, 3.0], power=2.5, num_rays=10)
    rays = source.emit_rays()
    assert all(ray.power == pytest.approx(0.25) for ray in rays)


def test_emit_rays_zero_power_allowed():
    source = PointSource([0.0, 0.0, 3.0], power=0.0, num_rays=8)
    rays = source.emit_rays()
    assert all(ray.power == 0.0 for ray in rays)
