import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import sympy as sp

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


def _editor(window):
    return {
        "selector": window.findChild(QtWidgets.QComboBox, "lensSelector"),
        "index": window.findChild(QtWidgets.QDoubleSpinBox, "doubleSpinBox"),
        "radius": window.findChild(QtWidgets.QDoubleSpinBox, "doubleSpinBox_2"),
        "s1": window.findChild(QtWidgets.QLineEdit, "lineEdit"),
        "s2": window.findChild(QtWidgets.QLineEdit, "lineEdit_2"),
        "buttons": window.findChild(QtWidgets.QDialogButtonBox, "buttonBox"),
    }


def _click(window, role):
    buttons = _editor(window)["buttons"]
    buttons.button(role).click()


def test_property_editor_disabled_without_stack(qt_app):
    window = OpticsMainWindow(make_display([], []))
    assert not _editor(window)["selector"].isEnabled()
    assert not _editor(window)["buttons"].isEnabled()
    window.close()


def test_load_stack_populates_lens_selector(qt_app, repo_root):
    window = OpticsMainWindow(make_display([], []))
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    assert editor["selector"].isEnabled()
    items = [editor["selector"].itemText(i) for i in range(editor["selector"].count())]
    assert items == ["lens1", "lens2"]
    assert editor["index"].value() == pytest.approx(1.5)
    assert editor["radius"].value() == pytest.approx(0.75)
    assert editor["s1"].text() == "-1 + sqrt(9 - (u**2) - (v**2))"
    assert editor["s2"].text() == "4.6 - sqrt(9 - (u**2) - (v**2))"
    window.close()


def test_switching_lens_repopulates_fields(qt_app, repo_root):
    window = OpticsMainWindow(make_display([], []))
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    editor["selector"].setCurrentText("lens2")
    assert editor["s1"].text() == "2.8 - sqrt(4 - (u**2) - (v**2))"
    assert editor["s2"].text() == "-1.3 + sqrt(4 - (u**2) - (v**2))"
    window.close()


def test_ok_applies_changes(qt_app, repo_root):
    window = OpticsMainWindow(make_display([], []))
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    editor["index"].setValue(1.8)
    editor["radius"].setValue(0.5)
    _click(window, QtWidgets.QDialogButtonBox.StandardButton.Ok)

    element = window.elements[0]
    assert element.material.refractive_index == pytest.approx(1.8)
    assert element.surface1.aperture.radius == pytest.approx(0.5)
    assert element.surface2.aperture.radius == pytest.approx(0.5)
    assert _mesh_count(window.canvas.view) >= 6
    assert "Updated properties for lens1" in window.statusBar().currentMessage()
    window.close()


def test_ok_applies_surface_expressions(qt_app, repo_root):
    window = OpticsMainWindow(make_display([], []))
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    editor["s1"].setText("0")
    _click(window, QtWidgets.QDialogButtonBox.StandardButton.Ok)

    assert str(window.elements[0].surface1.z_expr) == "0"
    assert window.elements[0].surface2.z_expr == sp.parse_expr("4.6 - sqrt(9 - (u**2) - (v**2))")
    window.close()


def test_cancel_reverts_edits(qt_app, repo_root):
    window = OpticsMainWindow(make_display([], []))
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    original = editor["s1"].text()
    editor["s1"].setText("not a valid expression")
    _click(window, QtWidgets.QDialogButtonBox.StandardButton.Cancel)

    assert editor["s1"].text() == original
    assert str(window.elements[0].surface1.z_expr) != "not a valid expression"
    window.close()


def test_invalid_expression_warns_and_keeps_scene(qt_app, repo_root, monkeypatch):
    window = OpticsMainWindow(make_display([], []))
    window.load_stack(repo_root / "tests" / "test_stack.json")

    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append(args)),
    )

    editor = _editor(window)
    editor["s1"].setText("1 +")
    _click(window, QtWidgets.QDialogButtonBox.StandardButton.Ok)

    assert warnings, "expected a warning dialog for an invalid expression"
    assert window.elements[0].surface1.z_expr == sp.parse_expr("-1 + sqrt(9 - (u**2) - (v**2))")
    assert "Updated properties" not in window.statusBar().currentMessage()
    window.close()
