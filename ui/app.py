from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtUiTools import QUiLoader

from vispy import app as vispy_app

import visualize

UI_PATH = Path(__file__).resolve().parent / "main.ui"


def make_display(elements, rays, x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), ray_length=6.0):
    canvas = visualize.create_canvas()
    view = canvas.central_widget.add_view(camera="turntable")
    visualize.build_scene(view, elements, rays, x_range=x_range, y_range=y_range, ray_length=ray_length)
    return canvas


class OpticsMainWindow(QtWidgets.QMainWindow):
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self._load_ui()
        self._embed_canvas()

    def _load_ui(self):
        loaded = QUiLoader().load(str(UI_PATH))
        self.setWindowTitle(loaded.windowTitle())
        self.setCentralWidget(loaded.centralWidget())
        self.setMenuWidget(loaded.menuWidget())
        self.setStatusBar(loaded.statusBar())

    def _embed_canvas(self):
        container = self.findChild(QtWidgets.QWidget, "openGLWidget")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas.native)

    @property
    def canvas(self):
        return self._canvas


def main():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    vispy_app.use_app("pyside6")

    canvas = make_display(visualize.default_stack(), visualize.default_rays())
    window = OpticsMainWindow(canvas)
    window.show()
    return QtWidgets.QApplication.instance().exec()


if __name__ == "__main__":
    raise SystemExit(main())
