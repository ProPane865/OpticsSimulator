"""Application entry point: main window wiring viewport and control panels."""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QDockWidget, QMenu
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtCore import Qt

from .models import Scene
from .viewport import Viewport
from .panels import SourcePanel, SurfacePanel, DetectorPanel


class MainWindow(QMainWindow):
    def __init__(self, preset="bi_convex", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optics Simulator")
        self.resize(1500, 920)

        self.scene = Scene(preset)
        self.viewport = Viewport()
        self.setCentralWidget(self.viewport.canvas.native)

        self.source_panel = SourcePanel(self.viewport, self.scene)
        self.surface_panel = SurfacePanel(self.viewport, self.scene)
        self.detector_panel = DetectorPanel(self.viewport, self.scene)

        self.addDockWidget(Qt.LeftDockWidgetArea,
                           self._dock("Source", self.source_panel))
        self.addDockWidget(Qt.RightDockWidgetArea,
                           self._dock("Surfaces", self.surface_panel))
        self.addDockWidget(Qt.RightDockWidgetArea,
                           self._dock("Detector", self.detector_panel))

        self._build_menu()
        self.viewport.set_scene(self.scene)

    def _dock(self, title, widget):
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        return dock

    def _build_menu(self):
        menubar = self.menuBar()
        sys_menu = menubar.addMenu("System")
        for name in ("bi_convex", "thin_lens", "ball_lens", "collimator"):
            act = sys_menu.addAction(name)
            act.triggered.connect(lambda _c=False, n=name: self._switch(n))
        sys_menu.addSeparator()
        exit_act = sys_menu.addAction("Exit")
        exit_act.triggered.connect(self.close)

    def _switch(self, preset):
        self.scene = Scene(preset)
        self.source_panel.scene = self.scene
        self.surface_panel.scene = self.scene
        self.detector_panel.scene = self.scene
        self.source_panel.sync_to_ui()
        self.surface_panel.refresh_all()
        self.detector_panel.sync_to_ui()
        self.viewport.set_scene(self.scene)


def main(argv=None):
    import sys
    from PySide6.QtWidgets import QApplication
    QSurfaceFormat.setDefaultFormat(_core_format())
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("Optics Simulator")
    win = MainWindow()
    win.show()
    return app.exec()


def _core_format():
    fmt = QSurfaceFormat()
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setMajorVersion(3)
    fmt.setMinorVersion(3)
    fmt.setSamples(4)
    return fmt


if __name__ == "__main__":
    main()
