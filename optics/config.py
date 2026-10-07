import json

import numpy as np

from .material import Material
from .refraction import RefractiveElement
from .surface import Surface
from .transform import Transform
from .vector import rotation_from_z


def surface_from_config(config, transform):
    return Surface(
        config["x"],
        config["y"],
        config["z"],
        transform=transform,
        u_range=config.get("u_range", (-2.0, 2.0)),
        v_range=config.get("v_range", (-2.0, 2.0)),
        aperture=config.get("aperture"),
    )


def refractive_element_from_config(
    config,
    orientation=np.array([0.0, 0.0, 1.0]),
    position=None,
):
    position = np.zeros(3) if position is None else np.asarray(position, dtype=float)
    transform = Transform(rotation_from_z(orientation), position)

    surface1 = surface_from_config(config["surface1"], transform)
    surface2 = surface_from_config(config["surface2"], transform)

    material = Material(refractive_index=float(config["material"]["refractive_index"]))

    return RefractiveElement(
        surface1=surface1,
        surface2=surface2,
        material=material,
    )


def refractive_element_from_json(
    schema,
    orientation=np.array([0.0, 0.0, 1.0]),
    position=None,
):
    return refractive_element_from_config(
        json.loads(schema),
        orientation=orientation,
        position=position,
    )
