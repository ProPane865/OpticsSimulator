"""Qt-free domain model for the desktop application.

Holds the current optical project (element configs, elements, light source,
rays and measurement planes) and the operations that mutate it. This module
has no PySide6/vispy dependencies so the naming, ordering and legacy-vs-
versioned file semantics can be reasoned about and unit-tested in isolation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from optics.config import stack_from_config
from optics.element import RefractiveElement
from optics.measurement_plane import MeasurementPlane, MeasurementResult, rays_to_arrays
from optics.primitive import Ray
from optics.project import (
    DEFAULT_SOURCE_CONFIG,
    build_project_document,
    parse_project,
)
from optics.source import PointSource
from optics.system import OpticalSystem

__all__ = [
    "DEFAULT_ELEMENT_CONFIG",
    "ProjectState",
    "classify_element_edit",
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


_ELEMENT_GEOMETRY_FIELDS = ("x", "y", "z", "u_range", "v_range", "aperture")
_ELEMENT_TRANSFORM_KEYS = ("position", "orientation")


def _only_index_changed(old_material: dict, new_material: dict) -> bool:
    if set(old_material) != set(new_material):
        return False
    for key in old_material:
        if key == "refractive_index":
            continue
        if old_material[key] != new_material[key]:
            return False
    return old_material.get("refractive_index") != new_material.get("refractive_index")


def classify_element_edit(
    old: dict[str, Any] | None, new: dict[str, Any] | None
) -> str:
    """Classify an element-config edit as ``none``, ``index``, ``transform`` or ``geometry``.

    The result lets the UI skip needless work: refractive-index-only edits do
    not change the rendered surface geometry, and a no-op edit needs no
    re-trace. Unknown or mixed changes conservatively report ``geometry``.
    """
    if old == new:
        return "none"
    old = old or {}
    new = new or {}

    geometry_changed = False
    for key in ("surface1", "surface2"):
        old_surface = old.get(key) or {}
        new_surface = new.get(key) or {}
        if any(old_surface.get(f) != new_surface.get(f) for f in _ELEMENT_GEOMETRY_FIELDS):
            geometry_changed = True

    transform_changed = any(old.get(k) != new.get(k) for k in _ELEMENT_TRANSFORM_KEYS)

    old_material = old.get("material") or {}
    new_material = new.get("material") or {}
    index_changed = old_material.get("refractive_index") != new_material.get("refractive_index")

    other_changed = old_material != new_material and not _only_index_changed(
        old_material, new_material
    )
    tracked = {"surface1", "surface2", "material", *_ELEMENT_TRANSFORM_KEYS}
    for key in set(old) | set(new):
        if key in tracked:
            continue
        if old.get(key) != new.get(key):
            other_changed = True

    if geometry_changed or other_changed:
        return "geometry"
    if transform_changed:
        return "transform"
    if index_changed:
        return "index"
    return "geometry"


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
        self._rays_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._planes: list[MeasurementPlane] = []
        self._plane_names: list[str] = []
        self._plane_results: list[MeasurementResult] = []
        self._system: OpticalSystem | None = None

    # -- read-only accessors -------------------------------------------------
    @property
    def schema(self) -> dict[str, Any] | None:
        return self._schema

    @property
    def elements(self) -> list[RefractiveElement]:
        return self._elements

    @property
    def system(self) -> OpticalSystem:
        """Persistent native system over the current element stack.

        Built lazily and dropped whenever the elements change, so the trace
        and measurement passes reuse one native ``TraceStack`` and
        configuration errors surface at (re)build, not per trace.
        """
        system = self._system
        if system is None:
            system = OpticalSystem(self._elements)
            self._system = system
        return system

    @property
    def source(self) -> PointSource:
        return self._source

    @property
    def rays(self) -> list[Ray]:
        return self._rays

    def rays_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The ray fan as contiguous ``(N, 3)`` origins/dirs and ``(N,)`` powers.

        Materialized lazily from :attr:`rays` and cached; the cache is dropped
        whenever the source/rays change, so per-refresh trace/measure passes can
        reuse the arrays without re-marshalling the ``Ray`` list every time.
        """
        arrays = self._rays_arrays
        if arrays is None:
            arrays = rays_to_arrays(self._rays)
            self._rays_arrays = arrays
        return arrays

    def measure_plane(self, plane: MeasurementPlane) -> MeasurementResult:
        """A fresh detector image for ``plane`` over the cached ray arrays.

        Array-based single-plane measurement (no per-``Ray`` marshalling);
        identical results to ``plane.measure(self.rays, self.system)``.
        """
        origins, dirs, powers = self.rays_arrays()
        return MeasurementPlane.measure_batch_arrays(
            [plane], origins, dirs, powers, self.system
        )[0]

    @property
    def planes(self) -> list[MeasurementPlane]:
        return self._planes

    @property
    def plane_names(self) -> list[str]:
        return self._plane_names

    @property
    def plane_results(self) -> list[MeasurementResult]:
        """Accumulated detector image per plane (parallel to :attr:`planes`)."""
        return self._plane_results

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
        self._system = None

    def replace_element(self, name: str, config: dict[str, Any], element: RefractiveElement) -> int:
        assert self._schema is not None
        index = list(self._schema.keys()).index(name)
        self._schema[name] = config
        self._elements[index] = element
        self._system = None
        return index

    def remove_element(self, name: str) -> None:
        assert self._schema is not None
        index = list(self._schema.keys()).index(name)
        del self._schema[name]
        self._elements.pop(index)
        self._system = None

    # -- source / rays -------------------------------------------------------
    def set_source(self, source: PointSource, rays: list[Ray]) -> None:
        self._source = source
        self._rays = rays
        self._rays_arrays = None

    # -- measurement plane operations ---------------------------------------
    def add_plane(self, name: str, plane: MeasurementPlane) -> MeasurementResult:
        result = self.measure_plane(plane)
        self._planes.append(plane)
        self._plane_names.append(name)
        self._plane_results.append(result)
        return result

    def replace_plane(
        self, index: int, plane: MeasurementPlane, result: MeasurementResult | None = None
    ) -> None:
        self._planes[index] = plane
        self._plane_results[index] = self.measure_plane(plane) if result is None else result

    def remove_plane(self, index: int) -> None:
        self._planes.pop(index)
        self._plane_names.pop(index)
        self._plane_results.pop(index)

    def set_plane_results(self, results: Sequence[MeasurementResult]) -> None:
        """Store the per-plane images produced by one combined trace."""
        self._plane_results = list(results)

    def measure_all(self) -> None:
        origins, dirs, powers = self.rays_arrays()
        self._plane_results = MeasurementPlane.measure_batch_arrays(
            self._planes, origins, dirs, powers, self.system
        )

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
        # Build the system up front: plane measurement needs the stack, and
        # this keeps construction-time validation inside the load.
        new_system = OpticalSystem(elements)
        ray_arrays = rays_to_arrays(new_rays)

        if planes is not None:
            new_planes = [plane for _, plane in planes]
            new_names: list[str] = []
            used: set[str] = set()
            for base_name, _ in planes:
                name = unique_name(base_name, used, prefix="plane")
                used.add(name)
                new_names.append(name)
            new_results = MeasurementPlane.measure_batch_arrays(
                new_planes, *ray_arrays, new_system
            )
            replaced = True
        else:
            new_results = MeasurementPlane.measure_batch_arrays(
                self._planes, *ray_arrays, new_system
            )
            new_planes = self._planes
            new_names = self._plane_names
            replaced = False

        self._schema = elements_config
        self._elements = elements
        self._system = new_system
        self._source = new_source
        self._rays = new_rays
        self._rays_arrays = None
        self._planes = new_planes
        self._plane_names = new_names
        self._plane_results = new_results
        return elements, replaced

    def to_document(self) -> dict[str, Any]:
        if self._schema is None:
            raise RuntimeError("No optical stack loaded to save")
        return build_project_document(self._schema, self._source, self._plane_names, self._planes)
