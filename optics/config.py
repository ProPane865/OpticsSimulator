import json

import numpy as np

from .material import Material
from .refraction import RefractiveElement
from .surface import Surface
from .transform import Transform
from .vector import rotation_from_z


def surface_from_config(config, transform):
    if not isinstance(config, dict):
        raise ValueError("Surface config must be an object")
    for key in ("x", "y", "z"):
        _require(config, key, "surface config")
    return Surface(
        config["x"],
        config["y"],
        config["z"],
        transform=transform,
        u_range=config.get("u_range", (-2.0, 2.0)),
        v_range=config.get("v_range", (-2.0, 2.0)),
        aperture=config.get("aperture"),
    )


def _require(mapping, key, context):
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"Missing required key {key!r} in {context}")
    return mapping[key]


def refractive_element_from_config(
    config,
    orientation=None,
    position=None,
):
    if not isinstance(config, dict):
        raise ValueError("Element config must be an object")
    if orientation is None:
        orientation = config.get("orientation", [0.0, 0.0, 1.0])
    if position is None:
        position = config.get("position", [0.0, 0.0, 0.0])
    orientation = np.asarray(orientation, dtype=float)
    position = np.asarray(position, dtype=float)
    transform = Transform(rotation_from_z(orientation), position)

    surface1 = surface_from_config(_require(config, "surface1", "element config"), transform)
    surface2 = surface_from_config(_require(config, "surface2", "element config"), transform)

    material_config = _require(config, "material", "element config")
    material = Material(
        refractive_index=float(_require(material_config, "refractive_index", "material config"))
    )

    return RefractiveElement(
        surface1=surface1,
        surface2=surface2,
        material=material,
    )


def refractive_element_from_json(
    schema,
    orientation=None,
    position=None,
):
    return refractive_element_from_config(
        json.loads(schema),
        orientation=orientation,
        position=position,
    )


def stack_from_config(stack_config):
    return [refractive_element_from_config(entry) for entry in stack_config.values()]
