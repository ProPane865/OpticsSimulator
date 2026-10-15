"""Standalone demo entry point: render the bundled doublet example without the UI."""

import json
from importlib.resources import files
from typing import Any

import numpy as np

from optics.config import stack_from_config
from optics.primitive.ray import Ray
from optics.refraction import RefractiveElement

from . import visualize as _visualize


def default_stack() -> list[RefractiveElement]:
    schema_text = (files("optics") / "examples" / "doublet.json").read_text()
    return stack_from_config(json.loads(schema_text))


def default_rays() -> list[Ray]:
    return [
        Ray(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0),
        Ray(np.array([0.5, 0.0, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0),
        Ray(np.array([0.0, 0.4, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0),
    ]


def visualize(
    elements: "RefractiveElement | list[RefractiveElement]",
    rays: "Ray | list[Ray]",
    x_range: "tuple[float, float]" = (-2.0, 2.0),
    y_range: "tuple[float, float]" = (-2.0, 2.0),
    ray_length: float = 3.0,
    show: bool = True,
) -> Any:
    canvas = _visualize.create_canvas()
    view = canvas.central_widget.add_view(camera="turntable")
    _visualize.build_scene(
        view, elements, rays, x_range=x_range, y_range=y_range, ray_length=ray_length
    )

    if show:
        canvas.show(run=True)
    return canvas


def main() -> None:
    visualize(default_stack(), default_rays(), ray_length=6.0)


if __name__ == "__main__":
    main()
