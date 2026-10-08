import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6 = pytest.importorskip("PySide6")

from PySide6 import QtWidgets

import rendering.visualize as visualize
from ui.app import OpticsMainWindow, make_display


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from vispy import app as vispy_app

    vispy_app.use_app("pyside6")
    yield app


def test_window_title(qt_app):
    window = OpticsMainWindow(make_display([], []))
    assert window.windowTitle() == "OpticsSimulator"
    window.close()


def test_canvas_embedded_in_opengl_widget(qt_app, make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    canvas = make_display([elem], visualize.default_rays())
    window = OpticsMainWindow(canvas)

    container = window.findChild(QtWidgets.QWidget, "openGLWidget")
    assert container is not None
    assert container.findChild(type(canvas.native)) is canvas.native
    assert window.canvas is canvas
    window.close()


def test_window_shows_with_scene(qt_app, make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    canvas = make_display([elem], visualize.default_rays())
    window = OpticsMainWindow(canvas)
    window.show()
    qt_app.processEvents()
    assert window.isVisible()
    window.close()
