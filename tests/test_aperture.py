import json

import numpy as np
import pytest

from optics.refraction import Ray, RefractiveElement
from optics.surface import Surface


APERTURE_RADIUS = 0.5


def make_surface(aperture_radius=APERTURE_RADIUS):
    """
    Upper unit hemisphere with a circular aperture centered on the z axis.
    """
    return Surface(
        "u",
        "v",
        "sqrt(1 - u**2 - v**2)",
        u_range=(-1.0, 1.0),
        v_range=(-1.0, 1.0),
        aperture={
            "radius": aperture_radius,
        },
    )


def make_element(aperture_radius=APERTURE_RADIUS, orientation=None):
    """
    Two hemispherical surfaces forming the existing test lens geometry.
    """
    schema = {
        "surface1": {
            "x": "u",
            "y": "v",
            "z": "sqrt(1 - u**2 - v**2)",
            "u_range": [-1.0, 1.0],
            "v_range": [-1.0, 1.0],
            "aperture": {
                "radius": aperture_radius,
            },
        },
        "surface2": {
            "x": "u",
            "y": "v",
            "z": "-sqrt(1 - u**2 - v**2)",
            "u_range": [-1.0, 1.0],
            "v_range": [-1.0, 1.0],
            "aperture": {
                "radius": aperture_radius,
            },
        },
        "material": {
            "refractive_index": 1.5,
        },
    }

    if orientation is None:
        return RefractiveElement(json.dumps(schema))

    return RefractiveElement(
        json.dumps(schema),
        np.asarray(orientation, dtype=float),
    )


def test_center_ray_intersects():
    """An on-axis ray must hit the surface."""
    surface = make_surface()

    ray = Ray(
        np.array([0.0, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
    )

    hit = surface.intersect(ray)

    assert hit is not None

    p, u, v = hit

    assert np.allclose(p, [0.0, 0.0, 1.0], atol=1e-8)
    assert np.isclose(u, 0.0, atol=1e-8)
    assert np.isclose(v, 0.0, atol=1e-8)


def test_ray_inside_aperture_intersects():
    """A ray strictly inside the aperture must intersect."""
    surface = make_surface()

    ray = Ray(
        np.array([0.25, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
    )

    hit = surface.intersect(ray)

    assert hit is not None

    p, _, _ = hit

    assert np.isclose(p[0], 0.25, atol=1e-8)
    assert np.isclose(p[1], 0.0, atol=1e-8)
    assert p[0] ** 2 + p[1] ** 2 < APERTURE_RADIUS ** 2


def test_ray_outside_aperture_does_not_intersect():
    """
    Critical regression test:
    the underlying sphere exists here, but the physical aperture does not.
    """
    surface = make_surface()

    ray = Ray(
        np.array([0.75, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
    )

    assert surface.intersect(ray) is None


def test_ray_just_inside_aperture_intersects():
    """Check behavior close to the aperture boundary."""
    surface = make_surface()

    ray = Ray(
        np.array([APERTURE_RADIUS - 1e-4, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
    )

    assert surface.intersect(ray) is not None


def test_ray_just_outside_aperture_does_not_intersect():
    """Check rejection immediately outside the aperture."""
    surface = make_surface()

    ray = Ray(
        np.array([APERTURE_RADIUS + 1e-4, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
    )

    assert surface.intersect(ray) is None


def test_ray_on_aperture_boundary_intersects():
    """
    The aperture definition uses <=, so the boundary itself belongs
    to the optical surface.
    """
    surface = make_surface()

    ray = Ray(
        np.array([APERTURE_RADIUS, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
    )

    hit = surface.intersect(ray)

    assert hit is not None

    p, _, _ = hit

    assert np.isclose(
        p[0] ** 2 + p[1] ** 2,
        APERTURE_RADIUS ** 2,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "x,y",
    [
        (0.75, 0.0),
        (-0.75, 0.0),
        (0.0, 0.75),
        (0.0, -0.75),
        (0.6, 0.6),
        (-0.6, 0.6),
        (0.6, -0.6),
        (-0.6, -0.6),
    ],
)
def test_aperture_rejects_all_directions(x, y):
    """
    Ensure aperture clipping isn't accidentally directional or dependent
    on one parameter axis.
    """
    surface = make_surface()

    ray = Ray(
        np.array([x, y, 2.0]),
        np.array([0.0, 0.0, -1.0]),
    )

    assert surface.intersect(ray) is None


def test_parameters_at_inside_aperture():
    """parameters_at() should recognize a physical point inside the aperture."""
    surface = make_surface()

    x = 0.25
    y = 0.10
    z = np.sqrt(1.0 - x**2 - y**2)

    uv = surface.parameters_at(
        np.array([x, y, z])
    )

    assert uv is not None

    u, v = uv

    assert np.isclose(u, x, atol=1e-8)
    assert np.isclose(v, y, atol=1e-8)


def test_parameters_at_outside_aperture_returns_none():
    """
    A point may lie exactly on the mathematical sphere while not belonging
    to the physical optical surface.
    """
    surface = make_surface()

    x = 0.75
    y = 0.0
    z = np.sqrt(1.0 - x**2)

    uv = surface.parameters_at(
        np.array([x, y, z])
    )

    assert uv is None


def test_mesh_vertices_are_inside_aperture():
    """No regular mesh vertex may lie outside the clear aperture."""
    surface = make_surface()

    verts, faces = surface.mesh(128)

    assert len(verts) > 0
    assert len(faces) > 0

    r2 = verts[:, 0] ** 2 + verts[:, 1] ** 2

    assert np.all(
        r2 <= APERTURE_RADIUS**2 + 1e-9
    )


def test_mesh_reaches_aperture_boundary():
    """
    Boundary interpolation should create vertices close to the actual
    aperture radius rather than stopping at the last interior grid sample.
    """
    surface = make_surface()

    verts, _ = surface.mesh(64)

    r = np.sqrt(
        verts[:, 0] ** 2
        + verts[:, 1] ** 2
    )

    assert np.isclose(
        np.max(r),
        APERTURE_RADIUS,
        atol=1e-6,
    )


def test_inside_aperture_ray_refracts():
    """A ray passing through the physical lens should produce an output ray."""
    element = make_element()

    ray = Ray(
        np.array([0.20, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
    )

    out = element.refract(ray)

    assert out is not None
    assert np.all(np.isfinite(out.origin))
    assert np.all(np.isfinite(out.direction))


def test_outside_aperture_ray_does_not_refract():
    """
    Critical integration regression:
    the ray intersects the underlying sphere but misses the actual lens.
    """
    element = make_element()

    ray = Ray(
        np.array([0.75, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
    )

    out = element.refract(ray)

    assert out is None


def test_rotated_element_on_axis_ray_refracts():
    """The aperture must rotate with the optical element."""
    orientation = np.array([1.0, 0.5, 1.0])
    orientation /= np.linalg.norm(orientation)

    element = make_element(
        orientation=orientation
    )

    ray = Ray(
        2.0 * orientation,
        -orientation,
    )

    out = element.refract(ray)

    assert out is not None
    assert np.all(np.isfinite(out.origin))
    assert np.all(np.isfinite(out.direction))


def test_rotated_element_rejects_ray_outside_local_aperture():
    """
    Verify that aperture clipping occurs in lens-local coordinates rather
    than fixed world XY coordinates.
    """
    orientation = np.array([1.0, 0.5, 1.0])
    orientation /= np.linalg.norm(orientation)

    element = make_element(
        orientation=orientation
    )

    # Local point displaced 0.75 along local x -- outside a radius-0.5
    # aperture.
    local_origin = np.array([0.75, 0.0, 2.0])
    local_direction = np.array([0.0, 0.0, -1.0])

    world_origin = element.r_matrix @ local_origin
    world_direction = element.r_matrix @ local_direction

    ray = Ray(
        world_origin,
        world_direction,
    )

    assert element.surface1.intersect(ray) is None
    assert element.refract(ray) is None