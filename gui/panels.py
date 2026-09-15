"""Control panels for source, surfaces and detector settings."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton,
    QLabel, QGridLayout, QVBoxLayout, QHBoxLayout,
)
from PySide6.QtCore import Qt, QTimer, QObject


class _Debouncer(QObject):
    """Triggers ``viewport.refresh()`` at most once per ``delay`` ms after edits."""

    def __init__(self, viewport, delay=250):
        super().__init__()
        self.viewport = viewport
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.setInterval(delay)
        self.timer.timeout.connect(viewport.refresh)

    def request(self):
        self.timer.start()


class SourcePanel(QWidget):
    def __init__(self, viewport, scene, parent=None):
        super().__init__(parent)
        self._debouncer = _Debouncer(viewport)
        self.scene = scene
        layout = QFormLayout(self)

        self.pattern_type = QComboBox()
        self.pattern_type.addItems(["gaussian", "disk", "uniform"])
        self.pattern_type.currentTextChanged.connect(self._emit)

        self.grid = QSpinBox()
        self.grid.setRange(8, 256)
        self.grid.setSingleStep(8)
        self.grid.valueChanged.connect(self._emit)

        self.pitch = QDoubleSpinBox()
        self.pitch.setRange(0.01, 100.0)
        self.pitch.setSingleStep(1.0)
        self.pitch.valueChanged.connect(self._emit)

        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.02, 0.9)
        self.sigma.setSingleStep(0.02)
        self.sigma.valueChanged.connect(self._emit)

        self.peak = QDoubleSpinBox()
        self.peak.setRange(0.0, 1e6)
        self.peak.setSingleStep(1.0)
        self.peak.valueChanged.connect(self._emit)

        self.origin_x = QDoubleSpinBox(); self.origin_x.setRange(-1e4, 1e4)
        self.origin_y = QDoubleSpinBox(); self.origin_y.setRange(-1e4, 1e4)
        self.origin_z = QDoubleSpinBox(); self.origin_z.setRange(-1e4, 1e4)
        for w in (self.origin_x, self.origin_y, self.origin_z):
            w.setSingleStep(1.0); w.valueChanged.connect(self._emit)

        layout.addRow("Pattern", self.pattern_type)
        layout.addRow("Grid", self.grid)
        layout.addRow("Pitch (mm)", self.pitch)
        layout.addRow("Sigma (fract)", self.sigma)
        layout.addRow("Peak", self.peak)
        layout.addRow("Origin X", self.origin_x)
        layout.addRow("Origin Y", self.origin_y)
        layout.addRow("Origin Z", self.origin_z)
        self.setLayout(layout)
        self.sync_to_ui()

    def _emit(self):
        self._sync_from_ui()
        self._debouncer.request()

    def sync_to_ui(self):
        widgets = [self.pattern_type, self.grid, self.pitch, self.sigma,
                   self.peak, self.origin_x, self.origin_y, self.origin_z]
        for w in widgets:
            w.blockSignals(True)
        try:
            s = self.scene
            self.pattern_type.setCurrentText(s.pattern_type)
            self.grid.setValue(int(s.grid))
            self.pitch.setValue(s.pitch)
            self.sigma.setValue(s.pattern_sigma)
            self.peak.setValue(s.pattern_peak)
            self.origin_x.setValue(s.source_origin[0])
            self.origin_y.setValue(s.source_origin[1])
            self.origin_z.setValue(s.source_origin[2])
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _sync_from_ui(self):
        s = self.scene
        s.pattern_type = self.pattern_type.currentText()
        s.grid = self.grid.value()
        s.pitch = self.pitch.value()
        s.pattern_sigma = self.sigma.value()
        s.pattern_peak = self.peak.value()
        s.source_origin = (self.origin_x.value(), self.origin_y.value(), self.origin_z.value())
        s._rebuild_source()


class SurfacePanel(QWidget):
    def __init__(self, viewport, scene, parent=None):
        QWidget.__init__(self, parent)
        self.viewport = viewport
        self.scene = scene
        layout = QVBoxLayout(self)
        self.container = QWidget()
        layout.addWidget(self.container)

        btns = QHBoxLayout()
        self.add_conic = QPushButton("Add Conic")
        self.add_conic.clicked.connect(self._add_conic)
        self.add_plane = QPushButton("Add Plane")
        self.add_plane.clicked.connect(self._add_plane)
        self.refresh = QPushButton("Trace")
        self.refresh.clicked.connect(viewport.refresh)
        self.autofocus = QPushButton("Auto-focus")
        self.autofocus.clicked.connect(self._autofocus)
        btns.addWidget(self.add_conic)
        btns.addWidget(self.add_plane)
        btns.addWidget(self.refresh)
        btns.addWidget(self.autofocus)
        layout.addLayout(btns)
        self._rows = []
        self.refresh_all()

    def refresh_all(self):
        for widget in self._rows:
            widget.setParent(None)
            widget.deleteLater()
        self._rows = []
        grid = QGridLayout(self.container)
        for i, s in enumerate(self.scene.surfaces):
            self._add_row(grid, i, s)
        self.container.setLayout(grid)

    def _add_row(self, grid, i, s):
        title = QLabel(f"{s['name']} ({s['kind']})")
        grid.addWidget(title, i, 0)

        z0 = QDoubleSpinBox(); z0.setRange(-1e4, 1e4); z0.setSingleStep(1.0)
        z0.setValue(s["z0"]); z0.valueChanged.connect(lambda _v, i=i: self._edit(i, "z0", _v))
        grid.addWidget(QLabel("z0"), i, 1)
        grid.addWidget(z0, i, 2)

        if s["kind"] == "conic":
            r = QDoubleSpinBox(); r.setRange(-1e4, 1e4); r.setSingleStep(1.0)
            r.setValue(s["radius"]); r.valueChanged.connect(lambda _v, i=i: self._edit(i, "radius", _v))
            c = QDoubleSpinBox(); c.setRange(-5, 5); c.setSingleStep(0.1)
            c.setValue(s["conicity"]); c.valueChanged.connect(lambda _v, i=i: self._edit(i, "conicity", _v))
            ap = QDoubleSpinBox(); ap.setRange(0, 1e4)
            ap.setValue(s["aperture"] if s["aperture"] is not None else -1)
            ap.valueChanged.connect(lambda _v, i=i: self._edit_aperture(i, _v))
            grid.addWidget(QLabel("R"), i, 3)
            grid.addWidget(r, i, 4)
            grid.addWidget(QLabel("k"), i, 5)
            grid.addWidget(c, i, 6)
            grid.addWidget(QLabel("aperture"), i, 7)
            grid.addWidget(ap, i, 8)
        else:
            n = QDoubleSpinBox(); n.setRange(1.0, 3.0); n.setSingleStep(0.01)
            n.setValue(s["n"]); n.valueChanged.connect(lambda _v, i=i: self._edit(i, "n", _v))
            grid.addWidget(QLabel("n"), i, 3)
            grid.addWidget(n, i, 4)
            ap = QDoubleSpinBox(); ap.setRange(0, 1e4)
            ap.setValue(s["aperture"] if s["aperture"] is not None else -1)
            ap.valueChanged.connect(lambda _v, i=i: self._edit_aperture(i, _v))
            grid.addWidget(QLabel("aperture"), i, 7)
            grid.addWidget(ap, i, 8)

        rm = QPushButton("X")
        rm.setFixedWidth(28)
        rm.clicked.connect(lambda _i=i: self._remove(i))
        grid.addWidget(rm, i, 9)
        self._rows.append(title)

    def _edit(self, i, key, v):
        self.scene.surfaces[i][key] = float(v)
        self.viewport.refresh()

    def _edit_aperture(self, i, v):
        self.scene.surfaces[i]["aperture"] = None if v < 0 else float(v)
        self.viewport.refresh()

    def _remove(self, i):
        self.scene.remove_surface(i)
        self.refresh_all()

    def _add_conic(self):
        self.scene.add_conic(radius=50.0, conicity=0.0, z0=0.0, n=1.5, aperture=45.0)
        self.refresh_all()

    def _add_plane(self):
        self.scene.add_plane(n=1.0, z0=0.0)
        self.refresh_all()

    def _autofocus(self):
        fz, _spot, _n = self.scene.spot_size()
        self.scene.detector_z = float(fz)
        self.viewport.refresh()


class DetectorPanel(QWidget):
    def __init__(self, viewport, scene, parent=None):
        QWidget.__init__(self, parent)
        self.scene = scene
        layout = QFormLayout(self)
        self.z = QDoubleSpinBox(); z = self.z
        z.setRange(-1e4, 1e4); z.setSingleStep(1.0)
        z.valueChanged.connect(lambda _v: (setattr(scene, "detector_z", float(_v)), viewport.refresh()))
        layout.addRow("Detector z", z)
        self.setLayout(layout)
        self.sync_to_ui()

    def sync_to_ui(self):
        self.z.setValue(self.scene.detector_z)
