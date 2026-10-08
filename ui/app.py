import json
from pathlib import Path

from PySide6 import QtGui, QtWidgets
from PySide6.QtUiTools import QUiLoader

from vispy import app as vispy_app

import rendering.visualize as visualize
from optics.config import stack_from_config

UI_PATH = Path(__file__).resolve().parent / "main.ui"


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
        self._load_ui()
        self._embed_canvas()
        self._connect_actions()

    def _load_ui(self):
        loaded = QUiLoader().load(str(UI_PATH))
        self._ui = loaded
        self.setWindowTitle(loaded.windowTitle())
        self.setCentralWidget(loaded.centralWidget())
        self.setMenuWidget(loaded.menuWidget())
        self.setStatusBar(loaded.statusBar())

    def _embed_canvas(self):
        container = self.findChild(QtWidgets.QWidget, "openGLWidget")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas.native)

    def _connect_actions(self):
        open_action = self._ui.findChild(QtGui.QAction, "actionOpen")
        if open_action is not None:
            open_action.triggered.connect(self._on_open)

    @property
    def canvas(self):
        return self._canvas

    def load_stack(self, path):
        path = Path(path)
        with open(path, "r") as f:
            schema = json.load(f)
        elements = stack_from_config(schema)

        visualize.clear_scene(self._view)
        visualize.build_scene(self._view, elements, self._rays, ray_length=6.0)

        self.statusBar().showMessage(f"Loaded {len(elements)} element(s) from {path.name}")
        return elements

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


def main():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    vispy_app.use_app("pyside6")

    stack = visualize.default_stack()
    rays = visualize.default_rays()
    canvas = make_display(stack, rays)
    window = OpticsMainWindow(canvas, rays=rays)
    window.show()
    return QtWidgets.QApplication.instance().exec()


if __name__ == "__main__":
    raise SystemExit(main())
