import copy
import json
from pathlib import Path

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtUiTools import QUiLoader

from vispy import app as vispy_app

import rendering.visualize as visualize
from optics.config import refractive_element_from_config, stack_from_config
from optics.source import POINT_ARRAY, SINGLE_POINT, PointSource

UI_PATH = Path(__file__).resolve().parent / "main.ui"

SOURCE_TYPE_BY_LABEL = {
    "Single Point": SINGLE_POINT,
    "Point Array": POINT_ARRAY,
}
LABEL_BY_SOURCE_TYPE = {value: key for key, value in SOURCE_TYPE_BY_LABEL.items()}

DEFAULT_SOURCE_CONFIG = {
    "position": [0.0, 0.0, 3.0],
    "power": 1.0,
    "num_rays": 64,
    "source_type": SINGLE_POINT,
    "cone_angle": 360.0,
}

DEFAULT_ELEMENT_CONFIG = {
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


def make_display(elements, rays, x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), ray_length=6.0):
    canvas = visualize.create_canvas()
    view = canvas.central_widget.add_view(camera="turntable")
    canvas.unfreeze()
    canvas.view = view
    canvas.freeze()
    visualize.build_scene(view, elements, rays, x_range=x_range, y_range=y_range, ray_length=ray_length)
    return canvas


class OpticsMainWindow(QtWidgets.QMainWindow):
    def __init__(self, canvas, rays=None, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self._view = canvas.view
        self._source = PointSource(**DEFAULT_SOURCE_CONFIG)
        self._source_nodes = []
        self._rays = rays if rays is not None else self._source.emit_rays()
        self._schema = None
        self._elements = []
        self._element_nodes = []
        self._ray_nodes = []
        self._load_ui()
        self._embed_canvas()
        self._connect_actions()
        self._connect_property_editor()
        self._connect_source_editor()
        self._refresh_source()

    def _load_ui(self):
        loaded = QUiLoader().load(str(UI_PATH))
        self._ui = loaded
        self.setWindowTitle(loaded.windowTitle())
        self.setCentralWidget(loaded.centralWidget())
        self.setMenuWidget(loaded.menuWidget())
        self.setStatusBar(loaded.statusBar())
        dock = loaded.findChild(QtWidgets.QDockWidget, "dockWidget")
        if dock is not None:
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _embed_canvas(self):
        container = self.findChild(QtWidgets.QWidget, "openGLWidget")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas.native)

    def _connect_actions(self):
        open_action = self._ui.findChild(QtGui.QAction, "actionOpen")
        if open_action is not None:
            open_action.triggered.connect(self._on_open)
        new_element_action = self._ui.findChild(QtGui.QAction, "actionOptical_Element")
        if new_element_action is not None:
            new_element_action.triggered.connect(self._on_new_element)
        save_as_action = self._ui.findChild(QtGui.QAction, "actionSave_As")
        if save_as_action is not None:
            save_as_action.triggered.connect(self._on_save_as)

    def _connect_property_editor(self):
        self._lens_selector = self.findChild(QtWidgets.QComboBox, "lensSelector")
        self._index_spin = self.findChild(QtWidgets.QDoubleSpinBox, "doubleSpinBox")
        self._radius_spin = self.findChild(QtWidgets.QDoubleSpinBox, "doubleSpinBox_2")
        self._surface1_edit = self.findChild(QtWidgets.QLineEdit, "lineEdit")
        self._surface2_edit = self.findChild(QtWidgets.QLineEdit, "lineEdit_2")
        self._button_box = self.findChild(QtWidgets.QDialogButtonBox, "buttonBox")

        self._index_spin.setRange(1.0, 3.0)
        self._index_spin.setSingleStep(0.01)
        self._radius_spin.setRange(0.01, 10.0)
        self._radius_spin.setSingleStep(0.05)

        self._lens_selector.currentIndexChanged.connect(self._populate_properties)
        self._button_box.accepted.connect(self._apply_properties)
        self._button_box.rejected.connect(self._populate_properties)
        self._set_property_editor_enabled(False)

    def _property_widgets(self):
        return (
            self._lens_selector,
            self._index_spin,
            self._radius_spin,
            self._surface1_edit,
            self._surface2_edit,
            self._button_box,
        )

    def _set_property_editor_enabled(self, enabled):
        for widget in self._property_widgets():
            widget.setEnabled(enabled)

    def _populate_lens_selector(self):
        self._lens_selector.blockSignals(True)
        self._lens_selector.clear()
        self._lens_selector.addItems(list(self._schema.keys()))
        self._lens_selector.blockSignals(False)
        has_lenses = self._lens_selector.count() > 0
        self._set_property_editor_enabled(has_lenses)
        if has_lenses:
            self._populate_properties()

    def _populate_properties(self, *_):
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

    def _apply_properties(self):
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

    def _retrace_rays(self):
        visualize.remove_nodes(self._ray_nodes)
        self._ray_nodes = []
        for ray in self._rays:
            nodes = visualize.draw_ray(self._view, self._elements, ray, ray_length=6.0)
            if nodes:
                self._ray_nodes.extend(nodes)

    def _connect_source_editor(self):
        self._source_type_combo = self.findChild(QtWidgets.QComboBox, "comboBox")
        self._power_spin = self.findChild(QtWidgets.QDoubleSpinBox, "doubleSpinBox_3")
        self._cone_angle_spin = self.findChild(QtWidgets.QDoubleSpinBox, "doubleSpinBox_4")
        self._num_rays_spin = self.findChild(QtWidgets.QSpinBox, "spinBox")
        self._source_x_spin = self.findChild(QtWidgets.QDoubleSpinBox, "sourceX")
        self._source_y_spin = self.findChild(QtWidgets.QDoubleSpinBox, "sourceY")
        self._source_z_spin = self.findChild(QtWidgets.QDoubleSpinBox, "sourceZ")
        self._source_button_box = self.findChild(QtWidgets.QDialogButtonBox, "sourceButtonBox")

        self._power_spin.setRange(0.0, 1000.0)
        self._power_spin.setSingleStep(0.1)
        self._cone_angle_spin.setRange(0.0, 360.0)
        self._cone_angle_spin.setSingleStep(1.0)
        self._num_rays_spin.setRange(1, 512)
        for spin in (self._source_x_spin, self._source_y_spin, self._source_z_spin):
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.1)

        point_array_item = self._source_type_combo.model().item(
            self._source_type_combo.findText("Point Array")
        )
        if point_array_item is not None:
            point_array_item.setEnabled(False)
            point_array_item.setToolTip("Point array sources are not implemented yet")

        self._source_button_box.accepted.connect(self._apply_source)
        self._source_button_box.rejected.connect(self._populate_source)
        self._populate_source()

    def _source_widgets(self):
        return (
            self._source_type_combo,
            self._power_spin,
            self._cone_angle_spin,
            self._num_rays_spin,
            self._source_x_spin,
            self._source_y_spin,
            self._source_z_spin,
        )

    def _populate_source(self, *_):
        self._source_type_combo.setCurrentText(LABEL_BY_SOURCE_TYPE.get(self._source.source_type, "Single Point"))
        self._power_spin.setValue(float(self._source.power))
        self._cone_angle_spin.setValue(float(self._source.cone_angle))
        self._num_rays_spin.setValue(int(self._source.num_rays))
        self._source_x_spin.setValue(float(self._source.position[0]))
        self._source_y_spin.setValue(float(self._source.position[1]))
        self._source_z_spin.setValue(float(self._source.position[2]))

    def _apply_source(self):
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

    def _refresh_source(self):
        visualize.remove_nodes(self._source_nodes)
        self._source_nodes = visualize.build_source_nodes(self._view, self._source)

    @property
    def canvas(self):
        return self._canvas

    @property
    def source(self):
        return self._source

    @property
    def rays(self):
        return list(self._rays)

    @property
    def elements(self):
        return list(self._elements)

    def load_stack(self, path):
        path = Path(path)
        with open(path, "r") as f:
            schema = json.load(f)
        self._schema = schema
        self._elements = stack_from_config(schema)
        elements = self._elements

        visualize.clear_scene(self._view)
        _, self._element_nodes, self._ray_nodes = visualize.build_scene(
            self._view, elements, self._rays, ray_length=6.0
        )
        self._refresh_source()

        self._populate_lens_selector()
        self.statusBar().showMessage(f"Loaded {len(elements)} element(s) from {path.name}")
        return elements

    def save_stack(self, path):
        if self._schema is None:
            raise RuntimeError("No optical stack loaded to save")
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self._schema, f, indent=4)
        self.statusBar().showMessage(f"Saved {len(self._schema)} element(s) to {path.name}")
        return path

    def _new_element_name(self):
        index = 1
        while f"lens{index}" in self._schema:
            index += 1
        return f"lens{index}"

    def _on_new_element(self):
        if self._schema is None:
            self._schema = {}
        name = self._new_element_name()
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

    def _on_open(self):
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

    def _on_save_as(self):
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
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".json")
        try:
            self.save_stack(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save Failed", f"Could not save optical stack:\n{exc}")


def main():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    vispy_app.use_app("pyside6")

    canvas = make_display([], [])
    window = OpticsMainWindow(canvas)
    window.load_stack(Path(__file__).resolve().parent.parent / "tests" / "test_stack.json")
    window.show()
    return QtWidgets.QApplication.instance().exec()


if __name__ == "__main__":
    raise SystemExit(main())
