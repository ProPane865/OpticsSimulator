import json
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


def test_file_menu_is_populated(qt_app):
    window = OpticsMainWindow(make_display([], []))
    menu_bar = window.findChild(QtWidgets.QMenuBar, "menubar")
    assert menu_bar is not None
    file_menu = menu_bar.findChild(QtWidgets.QMenu, "menuFile")
    assert file_menu is not None
    assert not file_menu.isEmpty()
    assert any(a.objectName() == "actionOpen" for a in file_menu.actions())
    window.close()


def test_window_shows_with_scene(qt_app, make_element):
    elem = make_element("sqrt(1 - (x**2) - (y**2))", "0")
    canvas = make_display([elem], visualize.default_rays())
    window = OpticsMainWindow(canvas)
    window.show()
    qt_app.processEvents()
    assert window.isVisible()
    window.close()


def _mesh_count(view):
    return sum(1 for node in view.scene.children if type(node).__name__ == "Mesh")


def test_load_stack_rebuilds_scene(qt_app, repo_root):
    canvas = make_display([], [])
    window = OpticsMainWindow(canvas)
    assert _mesh_count(canvas.view) == 0

    elements = window.load_stack(repo_root / "tests" / "test_stack.json")
    assert len(elements) == 2
    assert _mesh_count(canvas.view) >= 6
    assert "test_stack.json" in window.statusBar().currentMessage()
    window.close()


def test_load_stack_invalid_json_raises(qt_app, tmp_path):
    canvas = make_display([], [])
    window = OpticsMainWindow(canvas)
    bad = tmp_path / "bad_stack.json"
    bad.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        window.load_stack(bad)
    window.close()


def test_open_action_triggers_load(qt_app, repo_root, monkeypatch):
    canvas = make_display([], [])
    window = OpticsMainWindow(canvas)

    schema_path = repo_root / "tests" / "test_stack.json"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: (str(schema_path), "")),
    )

    file_menu = window.findChild(QtWidgets.QMenu, "menuFile")
    assert file_menu is not None
    matches = [a for a in file_menu.actions() if a.objectName() == "actionOpen"]
    assert matches, "actionOpen missing from File menu"
    matches[0].trigger()

    assert _mesh_count(canvas.view) >= 6
    assert "test_stack.json" in window.statusBar().currentMessage()
    window.close()
