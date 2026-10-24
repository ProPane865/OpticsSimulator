"""Standalone demo entry point: render the bundled doublet example without the UI."""

import json
from importlib.resources import files
from typing import Any

import numpy as np

from optics.config import stack_from_config
from optics.element import RefractiveElement
from optics.primitive import Ray

from . import visualize as _visualize


def default_stack() -> list[RefractiveElement]:
    schema_text = (files("optics") / "examples" / "doublet.json").read_text()
    return stack_from_config(json.loads(schema_text))


def default_rays() -> list[Ray]:
    return [
        Ray(np.array([0.0, 0.0, 30.0]), np.array([0.0, 0.0, -1.0]), 1.0),
        Ray(np.array([5.0, 0.0, 30.0]), np.array([0.0, 0.0, -1.0]), 1.0),
        Ray(np.array([0.0, 4.0, 30.0]), np.array([0.0, 0.0, -1.0]), 1.0),
    ]


def visualize(
    elements: list[RefractiveElement],
    rays: list[Ray],
    ray_length: float = _visualize.DEFAULT_RAY_LENGTH,
    show: bool = True,
) -> Any:
    canvas = _visualize.create_canvas()
    view = canvas.central_widget.add_view(camera="turntable")
    _visualize.build_scene(view, elements, rays, ray_length=ray_length)
    _visualize.frame_view(
        view, _visualize.scene_bounds(elements, ray_length=ray_length)
    )

    if show:
        canvas.show(run=True)
    return canvas


def main() -> None:
    visualize(default_stack(), default_rays())


if __name__ == "__main__":
    main()
