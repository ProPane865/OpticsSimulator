"""PySide6 desktop application: composition root for the optics simulator.

The window wires the Qt UI (``main.ui``) to the Qt-free domain
(:mod:`ui.model`), the vispy view (:mod:`ui.scene`) and the property editors
(:mod:`ui.editors`); it stays a thin orchestrator.
"""

import json
import threading
from collections.abc import Sequence
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, cast

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtUiTools import QUiLoader
from vispy import app as vispy_app

from optics.element import RefractiveElement
from optics.measurement_plane import MeasurementPlane, MeasurementResult
from optics.primitive import Ray
from optics.project import (
    DEFAULT_MEASUREMENT_CONFIG,
    DEFAULT_SOURCE_CONFIG,
    build_project_document,
    parse_project,
)
from optics.source import PointSource

from .editors import LABEL_BY_SOURCE_TYPE, SOURCE_TYPE_BY_LABEL, ElementEditor, PlaneEditor, SourceEditor
from .model import DEFAULT_ELEMENT_CONFIG, ProjectState
from .plane_view import PlaneView
from .scene import RAY_LENGTH, SceneController, make_display
from .widgets import find_child, run_validated

__all__ = [
    "DEFAULT_ELEMENT_CONFIG",
    "DEFAULT_MEASUREMENT_CONFIG",
    "DEFAULT_SOURCE_CONFIG",
    "LABEL_BY_SOURCE_TYPE",
    "OpticsMainWindow",
    "SOURCE_TYPE_BY_LABEL",
    "UI_PATH",
    "build_project_document",
    "main",
    "make_display",
    "parse_project",
]

UI_PATH = Path(__file__).resolve().parent / "main.ui"


class OpticsMainWindow(QtWidgets.QMainWindow):
    """Main window: owns the project state, scene and the three editors."""

    def __init__(
        self,
        canvas: Any,
        rays: Sequence[Ray] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._ui: QtWidgets.QMainWindow
        self._state = ProjectState(rays=rays)
        self._scene = SceneController(canvas)
        self._load_ui()
        self._embed_canvas(canvas)
        self._plane_view = PlaneView(find_child(self, QtWidgets.QWidget, "planeView", UI_PATH.name))
        self._element_editor = ElementEditor(
            self, self._state, self._scene, self._set_status, self._refresh_trace
        )
        self._source_editor = SourceEditor(
            self, self._state, self._scene, self._set_status, self._refresh_trace
        )
        self._plane_editor = PlaneEditor(
            self, self._state, self._scene, self._set_status, self._update_plane_view
        )
        self._connect_actions()
        self._scene.refresh_source(self._state.source)

    # -- UI construction -----------------------------------------------------
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

    def _embed_canvas(self, canvas: Any) -> None:
        container = find_child(self, QtWidgets.QWidget, "openGLWidget", UI_PATH.name)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas.native)

    def _connect_actions(self) -> None:
        find_child(self._ui, QtGui.QAction, "actionOpen", UI_PATH.name).triggered.connect(self._on_open)
        find_child(self._ui, QtGui.QAction, "actionOptical_Element", UI_PATH.name).triggered.connect(
            self._element_editor.add
        )
        find_child(self._ui, QtGui.QAction, "actionSave_As", UI_PATH.name).triggered.connect(self._on_save_as)
        find_child(self._ui, QtGui.QAction, "actionMeasurement_Plane", UI_PATH.name).triggered.connect(
            self._plane_editor.add
        )

    # -- orchestration shared by the editors --------------------------------
    def _refresh_trace(self) -> None:
        state = self._state
        system = state.system
        rays = state.rays
        planes = state.planes
        # When there are both rays and planes, one native walk over the
        # persistent stack serves rendering and every measurement plane;
        # otherwise use the separate measure/retrace paths.
        if rays and planes:
            origins, dirs, powers = state.rays_arrays()
            starts, ends, kinds, counts, us, vs, pws = system.trace_segments_and_planes(
                origins, dirs, powers, RAY_LENGTH, planes
            )
            results = []
            for q in range(len(planes)):
                result = planes[q].new_result()
                result.accumulate(us[q], vs[q], pws[q])
                results.append(result)
            state.set_plane_results(results)
            self._scene.retrace_from_segments(starts, ends, kinds, counts)
            self._scene.rebuild_planes(planes, results)
            self._update_plane_view()
            return
        state.measure_all()
        self._scene.retrace(system, rays)
        self._scene.rebuild_planes(planes, state.plane_results)
        self._update_plane_view()

    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def _update_plane_view(self) -> None:
        """Show the 2D colormap of the plane selected in the editor."""
        index = self._plane_editor.selected_index()
        results = self._state.plane_results
        result = results[index] if 0 <= index < len(results) else None
        self._plane_view.show_result(result)

    # -- public read-only accessors -----------------------------------------
    @property
    def canvas(self) -> Any:
        return self._scene.canvas

    @property
    def source(self) -> PointSource:
        return self._state.source

    @property
    def rays(self) -> "list[Ray]":
        return list(self._state.rays)

    @property
    def elements(self) -> "list[RefractiveElement]":
        return list(self._state.elements)

    @property
    def planes(self) -> "list[MeasurementPlane]":
        return list(self._state.planes)

    @property
    def plane_results(self) -> "list[MeasurementResult]":
        return list(self._state.plane_results)

    # -- project load / save -------------------------------------------------
    def load_stack(self, path: str | Path) -> "list[RefractiveElement]":
        path = Path(path)
        with open(path) as f:
            document = json.load(f)
        elements, replaced_planes = self._state.load_document(document)

        self._scene.full_rebuild(self._state.elements, self._state.rays)
        self._scene.refresh_source(self._state.source)
        self._scene.rebuild_planes(self._state.planes, self._state.plane_results)
        if replaced_planes:
            self._plane_editor.populate_selector()
        self._element_editor.populate_selector()
        self._update_plane_view()

        self._set_status(f"Loaded {len(elements)} element(s) from {path.name}")
        return elements

    def save_stack(self, path: str | Path) -> Path:
        document = self._state.to_document()
        path = Path(path)
        with open(path, "w") as f:
            json.dump(document, f, indent=4)
        self._set_status(f"Saved {len(document['elements'])} element(s) to {path.name}")
        return path

    # -- menu / dialog handlers ---------------------------------------------
    def _on_open(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Optical Stack",
            str(Path.home()),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        run_validated(
            self,
            lambda: self.load_stack(path),
            title="Open Failed",
            message="Could not load optical stack",
            critical=True,
        )

    def _on_save_as(self) -> None:
        if self._state.schema is None:
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
        run_validated(
            self,
            lambda: self.save_stack(save_path),
            title="Save Failed",
            message="Could not save optical stack",
            critical=True,
        )


def _warm_trace_kernels(elements: Sequence[RefractiveElement]) -> None:
    """Prime the batch trace/measure paths off the GUI thread.

    The native kernels need no compilation, but the first real trace still
    pays one-time setup (native elements for freshly loaded stacks);
    priming it here keeps the first add-element/apply-properties interaction
    responsive. Best-effort: any failure just leaves the work for the first
    interactive trace.
    """
    from optics.config import refractive_element_from_config
    from optics.system import OpticalSystem

    try:
        if not elements:
            elements = [refractive_element_from_config(DEFAULT_ELEMENT_CONFIG)]
        origins, dirs, powers = PointSource(**DEFAULT_SOURCE_CONFIG).emit_arrays()
        system = OpticalSystem(list(elements))
        system.trace_segments(origins, dirs, ray_length=6.0)
        plane = MeasurementPlane(**DEFAULT_MEASUREMENT_CONFIG)
        MeasurementPlane.measure_batch_arrays([plane], origins, dirs, powers, system)
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
