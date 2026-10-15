"""Qt-free domain model for the desktop application.

Holds the current optical project (element configs, elements, light source,
rays and measurement planes) and the operations that mutate it. This module
has no PySide6/vispy dependencies so the naming, ordering and legacy-vs-
versioned file semantics can be reasoned about and unit-tested in isolation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from optics.config import stack_from_config
from optics.measurement_plane import MeasurementPlane
from optics.primitive import Ray
from optics.project import (
    DEFAULT_SOURCE_CONFIG,
    build_project_document,
    parse_project,
)
from optics.source import PointSource
from optics.tracing import RefractiveElement

__all__ = [
    "DEFAULT_ELEMENT_CONFIG",
    "ProjectState",
    "unique_name",
]

DEFAULT_ELEMENT_CONFIG: dict[str, Any] = {
    "surface1": {
        "x": "u",
        "y": "v",
        "z": "-0.25",
        "u_range": [-2.0, 2.0],
        "v_range": [-2.0, 2.0],
        "aperture": {"radius": 0.75},
    },
    "surface2": {
        "x": "u",
        "y": "v",
        "z": "0.25",
        "u_range": [-2.0, 2.0],
        "v_range": [-2.0, 2.0],
        "aperture": {"radius": 0.75},
    },
    "material": {"refractive_index": 1.5},
}


def unique_name(base: str, used: Iterable[str], *, prefix: str = "plane") -> str:
    """Return a name not present in ``used``.

    A non-empty ``base`` is honored when free; otherwise the first free
    ``{prefix}{n}`` (n starting at 1) is returned. This unifies the previously
    duplicated interactive-add and file-load naming into one code path.
    """
    taken = set(used)
    if base and base not in taken:
        return base
    index = 1
    while f"{prefix}{index}" in taken:
        index += 1
    return f"{prefix}{index}"


class ProjectState:
    """Mutable project state plus the operations the UI performs on it."""

    def __init__(
        self,
        source: PointSource | None = None,
        rays: Iterable[Ray] | None = None,
    ) -> None:
        self._schema: dict[str, Any] | None = None
        self._elements: list[RefractiveElement] = []
        self._source: PointSource = source if source is not None else PointSource(**DEFAULT_SOURCE_CONFIG)
        self._rays: list[Ray] = list(rays) if rays is not None else list(self._source.emit_rays())
        self._planes: list[MeasurementPlane] = []
        self._plane_names: list[str] = []

    # -- read-only accessors -------------------------------------------------
    @property
    def schema(self) -> dict[str, Any] | None:
        return self._schema

    @property
    def elements(self) -> list[RefractiveElement]:
        return self._elements

    @property
    def source(self) -> PointSource:
        return self._source

    @property
    def rays(self) -> list[Ray]:
        return self._rays

    @property
    def planes(self) -> list[MeasurementPlane]:
        return self._planes

    @property
    def plane_names(self) -> list[str]:
        return self._plane_names

    # -- naming --------------------------------------------------------------
    def new_element_name(self) -> str:
        return unique_name("", (self._schema or {}).keys(), prefix="lens")

    def new_plane_name(self) -> str:
        return unique_name("", self._plane_names, prefix="plane")

    # -- element operations --------------------------------------------------
    def add_element(self, name: str, config: dict[str, Any], element: RefractiveElement) -> None:
        if self._schema is None:
            self._schema = {}
        self._schema[name] = config
        self._elements.append(element)

    def replace_element(self, name: str, config: dict[str, Any], element: RefractiveElement) -> int:
        assert self._schema is not None
        index = list(self._schema.keys()).index(name)
        self._schema[name] = config
        self._elements[index] = element
        return index

    # -- source / rays -------------------------------------------------------
    def set_source(self, source: PointSource, rays: list[Ray]) -> None:
        self._source = source
        self._rays = rays

    # -- measurement plane operations ---------------------------------------
    def add_plane(self, name: str, plane: MeasurementPlane) -> None:
        plane.measure(self._rays, self._elements)
        self._planes.append(plane)
        self._plane_names.append(name)

    def replace_plane(self, index: int, plane: MeasurementPlane) -> None:
        self._planes[index] = plane

    def remove_plane(self, index: int) -> None:
        self._planes.pop(index)
        self._plane_names.pop(index)

    def measure_all(self) -> None:
        for plane in self._planes:
            plane.measure(self._rays, self._elements)

    # -- document (de)serialization -----------------------------------------
    def load_document(self, document: object) -> tuple[list[RefractiveElement], bool]:
        """Apply a parsed project ``document`` to this state.

        Parsing and element construction run *before* any mutation so a bad
        file leaves the current state untouched (the callers re-raise). Returns
        ``(elements, replaced_planes)``; ``replaced_planes`` is ``False`` for
        legacy element-only files, where the current source and planes are kept.
        """
        elements_config, source, planes = parse_project(document)
        elements = stack_from_config(elements_config)

        new_source = source if source is not None else self._source
        new_rays = source.emit_rays() if source is not None else self._rays

        if planes is not None:
            new_planes: list[MeasurementPlane] = []
            new_names: list[str] = []
            used: set[str] = set()
            for base_name, plane in planes:
                name = unique_name(base_name, used, prefix="plane")
                used.add(name)
                plane.measure(new_rays, elements)
                new_planes.append(plane)
                new_names.append(name)
            replaced = True
        else:
            for plane in self._planes:
                plane.measure(new_rays, elements)
            new_planes = self._planes
            new_names = self._plane_names
            replaced = False

        self._schema = elements_config
        self._elements = elements
        self._source = new_source
        self._rays = new_rays
        self._planes = new_planes
        self._plane_names = new_names
        return elements, replaced

    def to_document(self) -> dict[str, Any]:
        if self._schema is None:
            raise RuntimeError("No optical stack loaded to save")
        return build_project_document(self._schema, self._source, self._plane_names, self._planes)
