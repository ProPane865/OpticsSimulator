import copy
import json
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtUiTools import QUiLoader

from vispy import app as vispy_app

import rendering.visualize as visualize
from optics.config import refractive_element_from_config, stack_from_config

UI_PATH = Path(__file__).resolve().parent / "main.ui"

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
        self._rays = rays if rays is not None else visualize.default_rays()
        self._schema = None
        self._elements = []
        self._element_nodes = []
        self._ray_nodes = []
        self._load_ui()
        self._embed_canvas()
        self._connect_actions()
        self._connect_property_editor()

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

    @property
    def canvas(self):
        return self._canvas

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

    rays = visualize.default_rays()
    canvas = make_display([], rays)
    window = OpticsMainWindow(canvas, rays=rays)
    window.load_stack(Path(__file__).resolve().parent.parent / "tests" / "test_stack.json")
    window.show()
    return QtWidgets.QApplication.instance().exec()


if __name__ == "__main__":
    raise SystemExit(main())
