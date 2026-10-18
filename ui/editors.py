"""Property-editor controllers.

Each editor is a controller over widgets that already exist in ``main.ui`` and
remain descendants of the main window (they are looked up by object name via
:func:`ui.widgets.find_child`, never re-parented). An editor reads/writes the
:class:`~ui.model.ProjectState`, requests redraws from the
:class:`~ui.scene.SceneController`, and posts status text, so the window itself
stays a thin composition root.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from PySide6 import QtGui, QtWidgets

from optics.config import refractive_element_from_config
from optics.measurement_plane import MeasurementPlane, MeasurementResult
from optics.project import DEFAULT_MEASUREMENT_CONFIG
from optics.source import POINT_ARRAY, SINGLE_POINT, PointSource

from .model import DEFAULT_ELEMENT_CONFIG, ProjectState, classify_element_edit
from .scene import SceneController
from .widgets import configure_spin, enable_widgets, find_child, run_validated

__all__ = [
    "LABEL_BY_SOURCE_TYPE",
    "SOURCE_TYPE_BY_LABEL",
    "ElementEditor",
    "PlaneEditor",
    "SourceEditor",
]

SOURCE_TYPE_BY_LABEL = {
    "Single Point": SINGLE_POINT,
    "Point Array": POINT_ARRAY,
}
LABEL_BY_SOURCE_TYPE = {value: key for key, value in SOURCE_TYPE_BY_LABEL.items()}


class ElementEditor:
    """Editor for a single refractive element's lens properties."""

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        state: ProjectState,
        scene: SceneController,
        set_status: Callable[[str], None],
        refresh_trace: Callable[[], None],
    ) -> None:
        self._window = window
        self._state = state
        self._scene = scene
        self._set_status = set_status
        self._refresh_trace = refresh_trace

        self._selector = find_child(window, QtWidgets.QComboBox, "lensSelector")
        self._index_spin = find_child(window, QtWidgets.QDoubleSpinBox, "refractiveIndexSpin")
        self._radius_spin = find_child(window, QtWidgets.QDoubleSpinBox, "apertureRadiusSpin")
        self._surface1_edit = find_child(window, QtWidgets.QLineEdit, "surface1Edit")
        self._surface2_edit = find_child(window, QtWidgets.QLineEdit, "surface2Edit")
        self._pos_spins: tuple[QtWidgets.QDoubleSpinBox, ...] = tuple(
            find_child(window, QtWidgets.QDoubleSpinBox, f"elementPos{axis}Spin") for axis in "XYZ"
        )
        self._normal_spins: tuple[QtWidgets.QDoubleSpinBox, ...] = tuple(
            find_child(window, QtWidgets.QDoubleSpinBox, f"elementNormal{axis}Spin") for axis in "XYZ"
        )
        self._button_box = find_child(window, QtWidgets.QDialogButtonBox, "lensButtonBox")

        configure_spin(self._index_spin, 1.0, 3.0, 0.01)
        configure_spin(self._radius_spin, 0.01, 10.0, 0.05)
        for spin in self._pos_spins:
            configure_spin(spin, -100.0, 100.0, 0.1)
        for spin in self._normal_spins:
            configure_spin(spin, -1.0, 1.0, 0.1, decimals=3)

        self._selector.currentIndexChanged.connect(self.populate_fields)
        self._button_box.accepted.connect(self.apply)
        self._button_box.rejected.connect(self.populate_fields)
        self.set_enabled(False)

    def _widgets(self) -> tuple[QtWidgets.QWidget, ...]:
        return (
            self._selector,
            self._index_spin,
            self._radius_spin,
            self._surface1_edit,
            self._surface2_edit,
            *self._pos_spins,
            *self._normal_spins,
            self._button_box,
        )

    def set_enabled(self, enabled: bool) -> None:
        enable_widgets(self._widgets(), enabled)

    def populate_selector(self) -> None:
        schema = self._state.schema or {}
        self._selector.blockSignals(True)
        self._selector.clear()
        self._selector.addItems(list(schema.keys()))
        self._selector.blockSignals(False)
        has_lenses = self._selector.count() > 0
        self.set_enabled(has_lenses)
        if has_lenses:
            self.populate_fields()

    def populate_fields(self, *_: object) -> None:
        schema = self._state.schema
        if not schema:
            return
        config = schema.get(self._selector.currentText())
        if config is None:
            return
        self._index_spin.setValue(float(config["material"]["refractive_index"]))
        aperture = config["surface1"].get("aperture") or {}
        if "radius" in aperture:
            self._radius_spin.setValue(float(aperture["radius"]))
        self._surface1_edit.setText(str(config["surface1"]["z"]))
        self._surface2_edit.setText(str(config["surface2"]["z"]))
        position = config.get("position", [0.0, 0.0, 0.0])
        orientation = config.get("orientation", [0.0, 0.0, 1.0])
        for spin, value in zip(self._pos_spins, position):
            spin.setValue(float(value))
        for spin, value in zip(self._normal_spins, orientation):
            spin.setValue(float(value))

    def apply(self) -> None:
        schema = self._state.schema
        if not schema:
            return
        name = self._selector.currentText()
        if name not in schema:
            return

        def build() -> tuple[Any, Any]:
            config = copy.deepcopy(schema[name])
            config["material"]["refractive_index"] = float(self._index_spin.value())
            config["surface1"]["z"] = self._surface1_edit.text()
            config["surface2"]["z"] = self._surface2_edit.text()
            config["position"] = [float(spin.value()) for spin in self._pos_spins]
            config["orientation"] = [float(spin.value()) for spin in self._normal_spins]
            radius = float(self._radius_spin.value())
            for key in ("surface1", "surface2"):
                aperture = config[key].get("aperture")
                if aperture is None:
                    aperture = {"type": "circular"}
                    config[key]["aperture"] = aperture
                aperture["radius"] = radius
            return config, refractive_element_from_config(config)

        result = run_validated(
            self._window, build, title="Invalid Lens Properties", message="Could not apply lens properties"
        )
        if result is None:
            return
        config, element = result
        old_config = (self._state.schema or {}).get(name)
        kind = classify_element_edit(old_config, config)
        index = self._state.replace_element(name, config, element)
        if kind == "none":
            self._set_status(f"No changes to {name}")
            return
        if kind in ("geometry", "transform"):
            self._scene.replace_element_nodes(index, element)
        self._refresh_trace()
        self._set_status(f"Updated properties for {name}")

    def select(self, name: str) -> None:
        self._selector.setCurrentText(name)
        self.populate_fields()

    def add(self) -> None:
        name = self._state.new_element_name()
        config = copy.deepcopy(DEFAULT_ELEMENT_CONFIG)
        element = run_validated(
            self._window,
            lambda: refractive_element_from_config(config),
            title="Invalid Element Properties",
            message="Could not add optical element",
        )
        if element is None:
            return
        self._state.add_element(name, config, element)
        self._scene.add_element_nodes(element)
        self._refresh_trace()
        self.populate_selector()
        self.select(name)
        self._set_status(f"Added {name}")


class SourceEditor:
    """Editor for the light source; enabled from startup (never needs a stack)."""

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        state: ProjectState,
        scene: SceneController,
        set_status: Callable[[str], None],
        refresh_trace: Callable[[], None],
    ) -> None:
        self._window = window
        self._state = state
        self._scene = scene
        self._set_status = set_status
        self._refresh_trace = refresh_trace

        self._type_combo = find_child(window, QtWidgets.QComboBox, "sourceTypeCombo")
        self._power_spin = find_child(window, QtWidgets.QDoubleSpinBox, "sourcePowerSpin")
        self._cone_spin = find_child(window, QtWidgets.QDoubleSpinBox, "coneAngleSpin")
        self._num_rays_spin = find_child(window, QtWidgets.QSpinBox, "numRaysSpin")
        self._x_spin = find_child(window, QtWidgets.QDoubleSpinBox, "sourceX")
        self._y_spin = find_child(window, QtWidgets.QDoubleSpinBox, "sourceY")
        self._z_spin = find_child(window, QtWidgets.QDoubleSpinBox, "sourceZ")
        self._button_box = find_child(window, QtWidgets.QDialogButtonBox, "sourceButtonBox")

        configure_spin(self._power_spin, 0.0, 1000.0, 0.1)
        configure_spin(self._cone_spin, 0.0, 360.0, 1.0)
        configure_spin(self._num_rays_spin, 1, 512)
        for spin in (self._x_spin, self._y_spin, self._z_spin):
            configure_spin(spin, -100.0, 100.0, 0.1)

        model = cast(QtGui.QStandardItemModel, self._type_combo.model())
        point_array_item = model.item(self._type_combo.findText("Point Array"))
        if point_array_item is not None:
            point_array_item.setEnabled(False)
            point_array_item.setToolTip("Point array sources are not implemented yet")

        self._button_box.accepted.connect(self.apply)
        self._button_box.rejected.connect(self.populate)
        self.populate()

    def populate(self, *_: object) -> None:
        source = self._state.source
        self._type_combo.setCurrentText(LABEL_BY_SOURCE_TYPE.get(source.source_type, "Single Point"))
        self._power_spin.setValue(float(source.power))
        self._cone_spin.setValue(float(source.cone_angle))
        self._num_rays_spin.setValue(int(source.num_rays))
        self._x_spin.setValue(float(source.position[0]))
        self._y_spin.setValue(float(source.position[1]))
        self._z_spin.setValue(float(source.position[2]))

    def apply(self) -> None:
        def build() -> tuple[PointSource, Any]:
            source = PointSource(
                position=np.array([self._x_spin.value(), self._y_spin.value(), self._z_spin.value()]),
                power=float(self._power_spin.value()),
                num_rays=int(self._num_rays_spin.value()),
                source_type=SOURCE_TYPE_BY_LABEL[self._type_combo.currentText()],
                cone_angle=float(self._cone_spin.value()),
            )
            return source, source.emit_rays()

        result = run_validated(
            self._window,
            build,
            title="Invalid Light Source Properties",
            message="Could not apply light source properties",
        )
        if result is None:
            return
        source, rays = result
        self._state.set_source(source, list(rays))
        self._scene.refresh_source(source)
        self._refresh_trace()
        self._set_status(f"Updated light source ({len(rays)} rays)")


class PlaneEditor:
    """Editor for adding, editing and removing measurement planes."""

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        state: ProjectState,
        scene: SceneController,
        set_status: Callable[[str], None],
    ) -> None:
        self._window = window
        self._state = state
        self._scene = scene
        self._set_status = set_status

        self._selector = find_child(window, QtWidgets.QComboBox, "planeSelectorCombo")
        self._remove_button = find_child(window, QtWidgets.QPushButton, "removePlaneButton")
        self._pos_spins: tuple[QtWidgets.QDoubleSpinBox, ...] = tuple(
            find_child(window, QtWidgets.QDoubleSpinBox, f"planePos{axis}Spin") for axis in "XYZ"
        )
        self._normal_spins: tuple[QtWidgets.QDoubleSpinBox, ...] = tuple(
            find_child(window, QtWidgets.QDoubleSpinBox, f"planeNormal{axis}Spin") for axis in "XYZ"
        )
        self._resolution_spins: tuple[QtWidgets.QSpinBox, ...] = (
            find_child(window, QtWidgets.QSpinBox, "planeResUSpin"),
            find_child(window, QtWidgets.QSpinBox, "planeResVSpin"),
        )
        self._button_box = find_child(window, QtWidgets.QDialogButtonBox, "planeButtonBox")

        for spin in self._pos_spins:
            configure_spin(spin, -100.0, 100.0, 0.1)
        for spin in self._normal_spins:
            configure_spin(spin, -1.0, 1.0, 0.1, decimals=3)
        for res_spin in self._resolution_spins:
            configure_spin(res_spin, 1, 512)

        self._selector.currentIndexChanged.connect(self.populate_fields)
        self._remove_button.clicked.connect(self.remove_current)
        self._button_box.accepted.connect(self.apply)
        self._button_box.rejected.connect(self.populate_fields)
        self.set_enabled(False)

    def _widgets(self) -> tuple[QtWidgets.QWidget, ...]:
        return (
            self._selector,
            self._remove_button,
            *self._pos_spins,
            *self._normal_spins,
            *self._resolution_spins,
            self._button_box,
        )

    def set_enabled(self, enabled: bool) -> None:
        enable_widgets(self._widgets(), enabled)

    def populate_selector(self) -> None:
        self._selector.blockSignals(True)
        self._selector.clear()
        self._selector.addItems(self._state.plane_names)
        self._selector.blockSignals(False)
        has_planes = self._selector.count() > 0
        self.set_enabled(has_planes)
        if has_planes:
            self.populate_fields()

    def populate_fields(self, *_: object) -> None:
        index = self._selector.currentIndex()
        planes = self._state.planes
        if index < 0 or index >= len(planes):
            return
        plane = planes[index]
        for spin, value in zip(self._pos_spins, plane.o):
            spin.setValue(float(value))
        for spin, value in zip(self._normal_spins, plane.n):
            spin.setValue(float(value))
        for res_spin, value in zip(self._resolution_spins, (plane.ru, plane.rv)):
            res_spin.setValue(int(value))

    def apply(self) -> None:
        index = self._selector.currentIndex()
        planes = self._state.planes
        if index < 0 or index >= len(planes):
            return
        old = planes[index]

        def build() -> tuple[MeasurementPlane, MeasurementResult]:
            updated = MeasurementPlane(
                origin=np.array([spin.value() for spin in self._pos_spins]),
                normal=np.array([spin.value() for spin in self._normal_spins]),
                u_range=(old.u_min, old.u_max),
                v_range=(old.v_min, old.v_max),
                resolution=[spin.value() for spin in self._resolution_spins],
            )
            result = updated.measure(self._state.rays, self._state.system)
            return updated, result

        applied = run_validated(
            self._window,
            build,
            title="Invalid Measurement Plane Properties",
            message="Could not apply measurement plane properties",
        )
        if applied is None:
            return
        updated, result = applied
        self._state.replace_plane(index, updated, result)
        self._scene.replace_plane_nodes(index, updated, result)
        self._set_status(f"Updated properties for {self._state.plane_names[index]}")

    def remove_current(self) -> None:
        index = self._selector.currentIndex()
        if index < 0 or index >= len(self._state.planes):
            return
        name = self._state.plane_names[index]
        self._scene.remove_plane_nodes(index)
        self._state.remove_plane(index)
        self.populate_selector()
        self._set_status(f"Removed {name}")

    def select(self, name: str) -> None:
        self._selector.setCurrentText(name)
        self.populate_fields()

    def add(self) -> None:
        config = copy.deepcopy(DEFAULT_MEASUREMENT_CONFIG)
        config["origin"][2] -= float(len(self._state.planes))
        plane = run_validated(
            self._window,
            lambda: MeasurementPlane(
                origin=np.asarray(config["origin"], dtype=float),
                normal=np.asarray(config["normal"], dtype=float),
                u_range=config["u_range"],
                v_range=config["v_range"],
                resolution=config["resolution"],
            ),
            title="Invalid Measurement Plane",
            message="Could not add measurement plane",
        )
        if plane is None:
            return
        name = self._state.new_plane_name()
        result = self._state.add_plane(name, plane)
        self._scene.add_plane_nodes(plane, result)
        self.populate_selector()
        self.select(name)
        self._set_status(f"Added {name}")
