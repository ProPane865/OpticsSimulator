"""Vispy scene/node bookkeeping for the desktop application.

The :class:`SceneController` owns the render canvas, the view and every node
list (elements, rays, source marker, measurement planes) so the window and
editors can request redraws without juggling vispy node objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import rendering.visualize as visualize
from optics.measurement_plane import MeasurementPlane
from optics.primitive import Ray
from optics.tracing import RefractiveElement

__all__ = ["SceneController", "make_display"]

RAY_LENGTH = 6.0


def make_display(
    elements: Sequence[RefractiveElement],
    rays: Sequence[Ray],
    x_range: tuple[float, float] = (-2.0, 2.0),
    y_range: tuple[float, float] = (-2.0, 2.0),
    ray_length: float = RAY_LENGTH,
) -> Any:
    canvas = visualize.create_canvas()
    view = canvas.central_widget.add_view(camera="turntable")
    canvas.unfreeze()
    canvas.view = view
    canvas.freeze()
    visualize.build_scene(view, elements, rays, x_range=x_range, y_range=y_range, ray_length=ray_length)
    return canvas


class SceneController:
    """Owns the vispy view and the node bookkeeping for every drawable kind."""

    def __init__(self, canvas: Any) -> None:
        self._canvas: Any = canvas
        self._view: Any = canvas.view
        self._source_nodes: list[Any] = []
        self._element_nodes: list[list[Any]] = []
        self._ray_nodes: list[Any] = []
        self._plane_nodes: list[list[Any]] = []

    @property
    def canvas(self) -> Any:
        return self._canvas

    @property
    def view(self) -> Any:
        return self._view

    # -- full / partial rebuilds --------------------------------------------
    def full_rebuild(
        self,
        elements: Sequence[RefractiveElement],
        rays: Sequence[Ray],
        ray_length: float = RAY_LENGTH,
    ) -> None:
        visualize.clear_scene(self._view)
        _, self._element_nodes, self._ray_nodes = visualize.build_scene(
            self._view, elements, rays, ray_length=ray_length
        )

    def refresh_source(self, source: Any) -> None:
        visualize.remove_nodes(self._source_nodes)
        self._source_nodes = visualize.build_source_nodes(self._view, source)

    def retrace(
        self,
        elements: Sequence[RefractiveElement],
        rays: Sequence[Ray],
        ray_length: float = RAY_LENGTH,
    ) -> None:
        visualize.remove_nodes(self._ray_nodes)
        self._ray_nodes = visualize.build_ray_nodes(self._view, elements, rays, ray_length=ray_length)

    # -- elements ------------------------------------------------------------
    def add_element_nodes(self, element: RefractiveElement) -> None:
        self._element_nodes.append(visualize.build_element_nodes(self._view, element))

    def replace_element_nodes(self, index: int, element: RefractiveElement) -> None:
        visualize.remove_nodes(self._element_nodes[index])
        self._element_nodes[index] = visualize.build_element_nodes(self._view, element)

    # -- measurement planes --------------------------------------------------
    def rebuild_planes(self, planes: Sequence[MeasurementPlane]) -> None:
        for nodes in self._plane_nodes:
            visualize.remove_nodes(nodes)
        self._plane_nodes = [visualize.build_measurement_nodes(self._view, plane) for plane in planes]

    def add_plane_nodes(self, plane: MeasurementPlane) -> None:
        self._plane_nodes.append(visualize.build_measurement_nodes(self._view, plane))

    def replace_plane_nodes(self, index: int, plane: MeasurementPlane) -> None:
        visualize.remove_nodes(self._plane_nodes[index])
        self._plane_nodes[index] = visualize.build_measurement_nodes(self._view, plane)

    def remove_plane_nodes(self, index: int) -> None:
        visualize.remove_nodes(self._plane_nodes.pop(index))
