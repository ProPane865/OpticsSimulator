import copy
import json
import threading
from collections.abc import Sequence
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtUiTools import QUiLoader
from vispy import app as vispy_app

import rendering.visualize as visualize
from optics.config import refractive_element_from_config, stack_from_config
from optics.measurement_plane import MeasurementPlane
from optics.primitive.ray import Ray
from optics.project import (
    DEFAULT_MEASUREMENT_CONFIG,
    DEFAULT_SOURCE_CONFIG,
    PROJECT_VERSION,
    build_project_document,
    parse_project,
    plane_from_config,
    plane_to_config,
    source_from_config,
    source_to_config,
)
from optics.refraction import RefractiveElement
from optics.source import POINT_ARRAY, SINGLE_POINT, PointSource

__all__ = [
    "DEFAULT_ELEMENT_CONFIG",
    "DEFAULT_MEASUREMENT_CONFIG",
    "DEFAULT_SOURCE_CONFIG",
    "LABEL_BY_SOURCE_TYPE",
    "OpticsMainWindow",
    "PROJECT_VERSION",
    "SOURCE_TYPE_BY_LABEL",
    "UI_PATH",
    "build_project_document",
    "main",
    "make_display",
    "parse_project",
    "plane_from_config",
    "plane_to_config",
    "source_from_config",
    "source_to_config",
]

UI_PATH = Path(__file__).resolve().parent / "main.ui"

_T = TypeVar("_T", bound=QtCore.QObject)

SOURCE_TYPE_BY_LABEL = {
    "Single Point": SINGLE_POINT,
    "Point Array": POINT_ARRAY,
}
LABEL_BY_SOURCE_TYPE = {value: key for key, value in SOURCE_TYPE_BY_LABEL.items()}

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


def _find_child(root: QtCore.QObject, widget_type: "type[_T]", name: str) -> "_T":
    child = root.findChild(widget_type, name)
    if child is None:
        raise RuntimeError(f"Missing widget {name!r} in {UI_PATH.name}")
    return child


def make_display(
    elements: Sequence[RefractiveElement],
    rays: Sequence[Ray],
    x_range: "tuple[float, float]" = (-2.0, 2.0),
    y_range: "tuple[float, float]" = (-2.0, 2.0),
    ray_length: float = 6.0,
) -> Any:
    canvas = visualize.create_canvas()
    view = canvas.central_widget.add_view(camera="turntable")
    canvas.unfreeze()
    canvas.view = view
    canvas.freeze()
    visualize.build_scene(view, elements, rays, x_range=x_range, y_range=y_range, ray_length=ray_length)
    return canvas


class OpticsMainWindow(QtWidgets.QMainWindow):
    _ui: QtWidgets.QMainWindow
    _lens_selector: QtWidgets.QComboBox
    _index_spin: QtWidgets.QDoubleSpinBox
    _radius_spin: QtWidgets.QDoubleSpinBox
    _surface1_edit: QtWidgets.QLineEdit
    _surface2_edit: QtWidgets.QLineEdit
    _element_pos_spins: "tuple[QtWidgets.QDoubleSpinBox, ...]"
    _element_normal_spins: "tuple[QtWidgets.QDoubleSpinBox, ...]"
    _button_box: QtWidgets.QDialogButtonBox
    _source_type_combo: QtWidgets.QComboBox
    _power_spin: QtWidgets.QDoubleSpinBox
    _cone_angle_spin: QtWidgets.QDoubleSpinBox
    _num_rays_spin: QtWidgets.QSpinBox
    _source_x_spin: QtWidgets.QDoubleSpinBox
    _source_y_spin: QtWidgets.QDoubleSpinBox
    _source_z_spin: QtWidgets.QDoubleSpinBox
    _source_button_box: QtWidgets.QDialogButtonBox
    _plane_selector: QtWidgets.QComboBox
    _plane_remove_button: QtWidgets.QPushButton
    _plane_pos_spins: "tuple[QtWidgets.QDoubleSpinBox, ...]"
    _plane_normal_spins: "tuple[QtWidgets.QDoubleSpinBox, ...]"
    _plane_resolution_spins: "tuple[QtWidgets.QSpinBox, ...]"
    _plane_button_box: QtWidgets.QDialogButtonBox

    def __init__(
        self,
        canvas: Any,
        rays: Sequence[Ray] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._canvas: Any = canvas
        self._view: Any = canvas.view
        self._source: PointSource = PointSource(**DEFAULT_SOURCE_CONFIG)
        self._source_nodes: list[Any] = []
        self._rays: list[Ray] = list(rays) if rays is not None else self._source.emit_rays()
        self._schema: dict[str, Any] | None = None
        self._elements: list[RefractiveElement] = []
        self._element_nodes: list[list[Any]] = []
        self._ray_nodes: list[Any] = []
        self._planes: list[MeasurementPlane] = []
        self._plane_nodes: list[list[Any]] = []
        self._plane_names: list[str] = []
        self._load_ui()
        self._embed_canvas()
        self._connect_actions()
        self._connect_property_editor()
        self._connect_source_editor()
        self._connect_plane_editor()
        self._refresh_source()

    def _load_ui(self) -> None:
        loaded = cast(QtWidgets.QMainWindow, QUiLoader().load(str(UI_PATH)))
        self._ui = loaded
        self.setWindowTitle(loaded.windowTitle())
        self.setCentralWidget(cast(QtWidgets.QWidget, loaded.centralWidget()))
        self.setMenuWidget(cast(QtWidgets.QWidget, loaded.menuWidget()))
        self.setStatusBar(loaded.statusBar())
        dock = loaded.findChild(QtWidgets.QDockWidget, "dockWidget")
        if dock is not None:
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _embed_canvas(self) -> None:
        container = _find_child(self, QtWidgets.QWidget, "openGLWidget")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas.native)

    def _connect_actions(self) -> None:
        _find_child(self._ui, QtGui.QAction, "actionOpen").triggered.connect(self._on_open)
        _find_child(self._ui, QtGui.QAction, "actionOptical_Element").triggered.connect(
            self._on_new_element
        )
        _find_child(self._ui, QtGui.QAction, "actionSave_As").triggered.connect(self._on_save_as)
        _find_child(self._ui, QtGui.QAction, "actionMeasurement_Plane").triggered.connect(
            self._on_new_measurement_plane
        )

    def _connect_property_editor(self) -> None:
        self._lens_selector = _find_child(self, QtWidgets.QComboBox, "lensSelector")
        self._index_spin = _find_child(self, QtWidgets.QDoubleSpinBox, "refractiveIndexSpin")
        self._radius_spin = _find_child(self, QtWidgets.QDoubleSpinBox, "apertureRadiusSpin")
        self._surface1_edit = _find_child(self, QtWidgets.QLineEdit, "surface1Edit")
        self._surface2_edit = _find_child(self, QtWidgets.QLineEdit, "surface2Edit")
        self._element_pos_spins = (
            _find_child(self, QtWidgets.QDoubleSpinBox, "elementPosXSpin"),
            _find_child(self, QtWidgets.QDoubleSpinBox, "elementPosYSpin"),
            _find_child(self, QtWidgets.QDoubleSpinBox, "elementPosZSpin"),
        )
        self._element_normal_spins = (
            _find_child(self, QtWidgets.QDoubleSpinBox, "elementNormalXSpin"),
            _find_child(self, QtWidgets.QDoubleSpinBox, "elementNormalYSpin"),
            _find_child(self, QtWidgets.QDoubleSpinBox, "elementNormalZSpin"),
        )
        self._button_box = _find_child(self, QtWidgets.QDialogButtonBox, "lensButtonBox")

        self._index_spin.setRange(1.0, 3.0)
        self._index_spin.setSingleStep(0.01)
        self._radius_spin.setRange(0.01, 10.0)
        self._radius_spin.setSingleStep(0.05)
        for spin in self._element_pos_spins:
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.1)
        for spin in self._element_normal_spins:
            spin.setRange(-1.0, 1.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(3)

        self._lens_selector.currentIndexChanged.connect(self._populate_properties)
        self._button_box.accepted.connect(self._apply_properties)
        self._button_box.rejected.connect(self._populate_properties)
        self._set_property_editor_enabled(False)

    def _property_widgets(self) -> "tuple[QtWidgets.QWidget, ...]":
        return (
            self._lens_selector,
            self._index_spin,
            self._radius_spin,
            self._surface1_edit,
            self._surface2_edit,
            *self._element_pos_spins,
            *self._element_normal_spins,
            self._button_box,
        )

    def _set_property_editor_enabled(self, enabled: bool) -> None:
        for widget in self._property_widgets():
            widget.setEnabled(enabled)

    def _populate_lens_selector(self) -> None:
        schema = self._schema or {}
        self._lens_selector.blockSignals(True)
        self._lens_selector.clear()
        self._lens_selector.addItems(list(schema.keys()))
        self._lens_selector.blockSignals(False)
        has_lenses = self._lens_selector.count() > 0
        self._set_property_editor_enabled(has_lenses)
        if has_lenses:
            self._populate_properties()

    def _populate_properties(self, *_: object) -> None:
        if not self._schema:
            return
        name = self._lens_selector.currentText()
        config = self._schema.get(name)
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
        for spin, value in zip(self._element_pos_spins, position):
            spin.setValue(float(value))
        for spin, value in zip(self._element_normal_spins, orientation):
            spin.setValue(float(value))

    def _apply_properties(self) -> None:
        if not self._schema:
            return
        name = self._lens_selector.currentText()
        if name not in self._schema:
            return
        try:
            config = copy.deepcopy(self._schema[name])
            config["material"]["refractive_index"] = float(self._index_spin.value())
            config["surface1"]["z"] = self._surface1_edit.text()
            config["surface2"]["z"] = self._surface2_edit.text()
            config["position"] = [float(spin.value()) for spin in self._element_pos_spins]
            config["orientation"] = [float(spin.value()) for spin in self._element_normal_spins]
            radius = float(self._radius_spin.value())
            for key in ("surface1", "surface2"):
                aperture = config[key].get("aperture")
                if aperture is None:
                    aperture = {"type": "circular"}
                    config[key]["aperture"] = aperture
                aperture["radius"] = radius
            element = refractive_element_from_config(config)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Lens Properties",
                f"Could not apply lens properties:\n{exc}",
            )
            return

        self._schema[name] = config
        index = list(self._schema.keys()).index(name)
        self._elements[index] = element

        visualize.remove_nodes(self._element_nodes[index])
        self._element_nodes[index] = visualize.build_element_nodes(self._view, element)
        self._retrace_rays()
        self.statusBar().showMessage(f"Updated properties for {name}")

    def _retrace_rays(self) -> None:
        visualize.remove_nodes(self._ray_nodes)
        self._ray_nodes = visualize.build_ray_nodes(
            self._view, self._elements, self._rays, ray_length=6.0
        )
        self._remeasure_planes()

    def _remeasure_planes(self) -> None:
        for index, plane in enumerate(self._planes):
            plane.measure(self._rays, self._elements)
            visualize.remove_nodes(self._plane_nodes[index])
            self._plane_nodes[index] = visualize.build_measurement_nodes(self._view, plane)

    def _connect_source_editor(self) -> None:
        self._source_type_combo = _find_child(self, QtWidgets.QComboBox, "sourceTypeCombo")
        self._power_spin = _find_child(self, QtWidgets.QDoubleSpinBox, "sourcePowerSpin")
        self._cone_angle_spin = _find_child(self, QtWidgets.QDoubleSpinBox, "coneAngleSpin")
        self._num_rays_spin = _find_child(self, QtWidgets.QSpinBox, "numRaysSpin")
        self._source_x_spin = _find_child(self, QtWidgets.QDoubleSpinBox, "sourceX")
        self._source_y_spin = _find_child(self, QtWidgets.QDoubleSpinBox, "sourceY")
        self._source_z_spin = _find_child(self, QtWidgets.QDoubleSpinBox, "sourceZ")
        self._source_button_box = _find_child(self, QtWidgets.QDialogButtonBox, "sourceButtonBox")

        self._power_spin.setRange(0.0, 1000.0)
        self._power_spin.setSingleStep(0.1)
        self._cone_angle_spin.setRange(0.0, 360.0)
        self._cone_angle_spin.setSingleStep(1.0)
        self._num_rays_spin.setRange(1, 512)
        for spin in (self._source_x_spin, self._source_y_spin, self._source_z_spin):
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.1)

        model = cast(QtGui.QStandardItemModel, self._source_type_combo.model())
        point_array_item = model.item(self._source_type_combo.findText("Point Array"))
        if point_array_item is not None:
            point_array_item.setEnabled(False)
            point_array_item.setToolTip("Point array sources are not implemented yet")

        self._source_button_box.accepted.connect(self._apply_source)
        self._source_button_box.rejected.connect(self._populate_source)
        self._populate_source()

    def _source_widgets(self) -> "tuple[QtWidgets.QWidget, ...]":
        return (
            self._source_type_combo,
            self._power_spin,
            self._cone_angle_spin,
            self._num_rays_spin,
            self._source_x_spin,
            self._source_y_spin,
            self._source_z_spin,
        )

    def _populate_source(self, *_: object) -> None:
        self._source_type_combo.setCurrentText(
            LABEL_BY_SOURCE_TYPE.get(self._source.source_type, "Single Point")
        )
        self._power_spin.setValue(float(self._source.power))
        self._cone_angle_spin.setValue(float(self._source.cone_angle))
        self._num_rays_spin.setValue(int(self._source.num_rays))
        self._source_x_spin.setValue(float(self._source.position[0]))
        self._source_y_spin.setValue(float(self._source.position[1]))
        self._source_z_spin.setValue(float(self._source.position[2]))

    def _apply_source(self) -> None:
        try:
            source = PointSource(
                position=np.array([
                    self._source_x_spin.value(),
                    self._source_y_spin.value(),
                    self._source_z_spin.value(),
                ]),
                power=float(self._power_spin.value()),
                num_rays=int(self._num_rays_spin.value()),
                source_type=SOURCE_TYPE_BY_LABEL[self._source_type_combo.currentText()],
                cone_angle=float(self._cone_angle_spin.value()),
            )
            rays = source.emit_rays()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Light Source Properties",
                f"Could not apply light source properties:\n{exc}",
            )
            return

        self._source = source
        self._rays = rays
        self._refresh_source()
        self._retrace_rays()
        self.statusBar().showMessage(f"Updated light source ({len(rays)} rays)")

    def _refresh_source(self) -> None:
        visualize.remove_nodes(self._source_nodes)
        self._source_nodes = visualize.build_source_nodes(self._view, self._source)

    def _connect_plane_editor(self) -> None:
        self._plane_selector = _find_child(self, QtWidgets.QComboBox, "planeSelectorCombo")
        self._plane_remove_button = _find_child(self, QtWidgets.QPushButton, "removePlaneButton")
        self._plane_pos_spins = (
            _find_child(self, QtWidgets.QDoubleSpinBox, "planePosXSpin"),
            _find_child(self, QtWidgets.QDoubleSpinBox, "planePosYSpin"),
            _find_child(self, QtWidgets.QDoubleSpinBox, "planePosZSpin"),
        )
        self._plane_normal_spins = (
            _find_child(self, QtWidgets.QDoubleSpinBox, "planeNormalXSpin"),
            _find_child(self, QtWidgets.QDoubleSpinBox, "planeNormalYSpin"),
            _find_child(self, QtWidgets.QDoubleSpinBox, "planeNormalZSpin"),
        )
        self._plane_resolution_spins = (
            _find_child(self, QtWidgets.QSpinBox, "planeResUSpin"),
            _find_child(self, QtWidgets.QSpinBox, "planeResVSpin"),
        )
        self._plane_button_box = _find_child(self, QtWidgets.QDialogButtonBox, "planeButtonBox")

        for spin in self._plane_pos_spins:
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.1)
        for spin in self._plane_normal_spins:
            spin.setRange(-1.0, 1.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(3)
        for res_spin in self._plane_resolution_spins:
            res_spin.setRange(1, 512)

        self._plane_selector.currentIndexChanged.connect(self._populate_plane_properties)
        self._plane_remove_button.clicked.connect(self._remove_plane)
        self._plane_button_box.accepted.connect(self._apply_plane_properties)
        self._plane_button_box.rejected.connect(self._populate_plane_properties)
        self._set_plane_editor_enabled(False)

    def _plane_editor_widgets(self) -> "tuple[QtWidgets.QWidget, ...]":
        return (
            self._plane_selector,
            self._plane_remove_button,
            *self._plane_pos_spins,
            *self._plane_normal_spins,
            *self._plane_resolution_spins,
            self._plane_button_box,
        )

    def _set_plane_editor_enabled(self, enabled: bool) -> None:
        for widget in self._plane_editor_widgets():
            widget.setEnabled(enabled)

    def _new_plane_name(self) -> str:
        index = 1
        while f"plane{index}" in self._plane_names:
            index += 1
        return f"plane{index}"

    def _populate_plane_selector(self) -> None:
        self._plane_selector.blockSignals(True)
        self._plane_selector.clear()
        self._plane_selector.addItems(self._plane_names)
        self._plane_selector.blockSignals(False)
        has_planes = self._plane_selector.count() > 0
        self._set_plane_editor_enabled(has_planes)
        if has_planes:
            self._populate_plane_properties()

    def _populate_plane_properties(self, *_: object) -> None:
        index = self._plane_selector.currentIndex()
        if index < 0 or index >= len(self._planes):
            return
        plane = self._planes[index]
        for spin, value in zip(self._plane_pos_spins, plane.o):
            spin.setValue(float(value))
        for spin, value in zip(self._plane_normal_spins, plane.n):
            spin.setValue(float(value))
        for res_spin, value in zip(self._plane_resolution_spins, (plane.ru, plane.rv)):
            res_spin.setValue(int(value))

    def _apply_plane_properties(self) -> None:
        index = self._plane_selector.currentIndex()
        if index < 0 or index >= len(self._planes):
            return
        plane = self._planes[index]
        try:
            updated = MeasurementPlane(
                origin=np.array([spin.value() for spin in self._plane_pos_spins]),
                normal=np.array([spin.value() for spin in self._plane_normal_spins]),
                u_range=(plane.u_min, plane.u_max),
                v_range=(plane.v_min, plane.v_max),
                resolution=[spin.value() for spin in self._plane_resolution_spins],
            )
            updated.measure(self._rays, self._elements)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Measurement Plane Properties",
                f"Could not apply measurement plane properties:\n{exc}",
            )
            return

        visualize.remove_nodes(self._plane_nodes[index])
        self._planes[index] = updated
        self._plane_nodes[index] = visualize.build_measurement_nodes(self._view, updated)
        self.statusBar().showMessage(f"Updated properties for {self._plane_names[index]}")

    def _remove_plane(self) -> None:
        index = self._plane_selector.currentIndex()
        if index < 0 or index >= len(self._planes):
            return
        name = self._plane_names[index]
        visualize.remove_nodes(self._plane_nodes.pop(index))
        self._planes.pop(index)
        self._plane_names.pop(index)
        self._populate_plane_selector()
        self.statusBar().showMessage(f"Removed {name}")

    @property
    def canvas(self) -> Any:
        return self._canvas

    @property
    def source(self) -> PointSource:
        return self._source

    @property
    def rays(self) -> list[Ray]:
        return list(self._rays)

    @property
    def elements(self) -> list[RefractiveElement]:
        return list(self._elements)

    @property
    def planes(self) -> list[MeasurementPlane]:
        return list(self._planes)

    @staticmethod
    def _unique_plane_name(base: str, used: set[str]) -> str:
        if base and base not in used:
            return base
        index = 1
        while f"plane{index}" in used:
            index += 1
        return f"plane{index}"

    def load_stack(self, path: str | Path) -> list[RefractiveElement]:
        path = Path(path)
        with open(path) as f:
            document = json.load(f)
        elements_config, source, planes = parse_project(document)
        elements = stack_from_config(elements_config)

        self._schema = elements_config
        self._elements = elements
        if source is not None:
            self._source = source
            self._rays = source.emit_rays()

        visualize.clear_scene(self._view)
        _, self._element_nodes, self._ray_nodes = visualize.build_scene(
            self._view, elements, self._rays, ray_length=6.0
        )
        self._refresh_source()

        if planes is not None:
            self._planes = []
            self._plane_names = []
            self._plane_nodes = []
            used: set[str] = set()
            for base_name, plane in planes:
                name = self._unique_plane_name(base_name, used)
                used.add(name)
                plane.measure(self._rays, elements)
                self._planes.append(plane)
                self._plane_names.append(name)
                self._plane_nodes.append(
                    visualize.build_measurement_nodes(self._view, plane)
                )
            self._populate_plane_selector()
        else:
            self._remeasure_planes()

        self._populate_lens_selector()
        self.statusBar().showMessage(f"Loaded {len(elements)} element(s) from {path.name}")
        return elements

    def save_stack(self, path: str | Path) -> Path:
        if self._schema is None:
            raise RuntimeError("No optical stack loaded to save")
        path = Path(path)
        document = build_project_document(
            self._schema, self._source, self._plane_names, self._planes
        )
        with open(path, "w") as f:
            json.dump(document, f, indent=4)
        self.statusBar().showMessage(f"Saved {len(self._schema)} element(s) to {path.name}")
        return path

    @staticmethod
    def _new_element_name(schema: dict[str, Any]) -> str:
        index = 1
        while f"lens{index}" in schema:
            index += 1
        return f"lens{index}"

    def _on_new_element(self) -> None:
        if self._schema is None:
            self._schema = {}
        name = self._new_element_name(self._schema)
        config = copy.deepcopy(DEFAULT_ELEMENT_CONFIG)
        try:
            element = refractive_element_from_config(config)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Element Properties",
                f"Could not add optical element:\n{exc}",
            )
            return

        self._schema[name] = config
        self._elements.append(element)
        self._element_nodes.append(visualize.build_element_nodes(self._view, element))
        self._retrace_rays()

        self._populate_lens_selector()
        self._lens_selector.setCurrentText(name)
        self._populate_properties()
        self.statusBar().showMessage(f"Added {name}")

    def _on_new_measurement_plane(self) -> None:
        config = copy.deepcopy(DEFAULT_MEASUREMENT_CONFIG)
        config["origin"][2] -= float(len(self._planes))
        try:
            plane = MeasurementPlane(
                origin=np.asarray(config["origin"], dtype=float),
                normal=np.asarray(config["normal"], dtype=float),
                u_range=config["u_range"],
                v_range=config["v_range"],
                resolution=config["resolution"],
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Measurement Plane",
                f"Could not add measurement plane:\n{exc}",
            )
            return

        name = self._new_plane_name()
        plane.measure(self._rays, self._elements)
        self._planes.append(plane)
        self._plane_names.append(name)
        self._plane_nodes.append(
            visualize.build_measurement_nodes(self._view, plane)
        )

        self._populate_plane_selector()
        self._plane_selector.setCurrentText(name)
        self._populate_plane_properties()
        self.statusBar().showMessage(f"Added {name}")

    def _on_open(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Optical Stack",
            str(Path.home()),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            self.load_stack(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open Failed", f"Could not load optical stack:\n{exc}")

    def _on_save_as(self) -> None:
        if self._schema is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Nothing to Save",
                "No optical stack is loaded.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Optical Stack",
            str(Path.home()),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        save_path = Path(path)
        if not save_path.suffix:
            save_path = save_path.with_suffix(".json")
        try:
            self.save_stack(save_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save Failed", f"Could not save optical stack:\n{exc}")


def _warm_trace_kernels(elements: Sequence[RefractiveElement]) -> None:
    """Precompile the fixed-capacity Numba stack kernels off the GUI thread.

    The trace/measure kernels compile once per capacity, not per element
    count, so priming them once after startup keeps the first
    add-element/apply-properties interaction from paying the compile cost
    synchronously. Best-effort: any failure just leaves on-demand compiles.
    """
    from optics.refraction import trace_segments_batch

    try:
        if not elements:
            elements = [refractive_element_from_config(DEFAULT_ELEMENT_CONFIG)]
        rays = PointSource(**DEFAULT_SOURCE_CONFIG).emit_rays()
        origins = np.asarray([r.origin for r in rays], dtype=float)
        dirs = np.asarray([r.direction for r in rays], dtype=float)
        trace_segments_batch(elements, origins, dirs, ray_length=6.0)
        plane = MeasurementPlane(**DEFAULT_MEASUREMENT_CONFIG)
        plane.measure(rays, list(elements))
    except Exception:
        pass


def main() -> int:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    vispy_app.use_app("pyside6")

    canvas = make_display([], [])
    window = OpticsMainWindow(canvas)
    with as_file(files("optics") / "examples" / "doublet.json") as example_path:
        window.load_stack(example_path)
    window.show()
    threading.Thread(
        target=_warm_trace_kernels, args=(window.elements,), daemon=True
    ).start()
    return qapp.exec()


if __name__ == "__main__":
    raise SystemExit(main())
