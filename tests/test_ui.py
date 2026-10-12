import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
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
    assert _mesh_count(canvas.view) == 1

    elements = window.load_stack(repo_root / "tests" / "test_stack.json")
    assert len(elements) == 2
    assert _mesh_count(canvas.view) >= 7
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
        "index": window.findChild(QtWidgets.QDoubleSpinBox, "refractiveIndexSpin"),
        "radius": window.findChild(QtWidgets.QDoubleSpinBox, "apertureRadiusSpin"),
        "s1": window.findChild(QtWidgets.QLineEdit, "surface1Edit"),
        "s2": window.findChild(QtWidgets.QLineEdit, "surface2Edit"),
        "buttons": window.findChild(QtWidgets.QDialogButtonBox, "lensButtonBox"),
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


def _trigger_new_element(window):
    menu = window.findChild(QtWidgets.QMenu, "menuNew")
    assert menu is not None
    matches = [a for a in menu.actions() if a.objectName() == "actionOptical_Element"]
    assert matches, "actionOptical_Element missing from New menu"
    matches[0].trigger()


def test_new_element_action_adds_to_stack(qt_app, repo_root):
    canvas = make_display([], [])
    window = OpticsMainWindow(canvas)
    window.load_stack(repo_root / "tests" / "test_stack.json")
    meshes_before = _mesh_count(canvas.view)

    _trigger_new_element(window)

    assert len(window.elements) == 3
    editor = _editor(window)
    items = [editor["selector"].itemText(i) for i in range(editor["selector"].count())]
    assert items == ["lens1", "lens2", "lens3"]
    assert editor["selector"].currentText() == "lens3"
    assert editor["selector"].isEnabled()
    assert _mesh_count(canvas.view) > meshes_before
    assert "Added lens3" in window.statusBar().currentMessage()
    window.close()


def test_new_element_on_empty_stack(qt_app):
    window = OpticsMainWindow(make_display([], []))
    _trigger_new_element(window)

    assert len(window.elements) == 1
    editor = _editor(window)
    assert editor["selector"].currentText() == "lens1"
    assert editor["selector"].isEnabled()
    assert "Added lens1" in window.statusBar().currentMessage()
    window.close()


def test_new_element_names_increment(qt_app):
    window = OpticsMainWindow(make_display([], []))
    _trigger_new_element(window)
    _trigger_new_element(window)

    assert len(window.elements) == 2
    items = [_editor(window)["selector"].itemText(i) for i in range(_editor(window)["selector"].count())]
    assert items == ["lens1", "lens2"]
    window.close()


def test_new_element_default_geometry(qt_app):
    window = OpticsMainWindow(make_display([], []))
    _trigger_new_element(window)

    element = window.elements[-1]
    assert float(element.surface2.z_expr - element.surface1.z_expr) == pytest.approx(0.5)
    assert element.material.refractive_index == pytest.approx(1.5)
    assert element.surface1.aperture.radius == pytest.approx(0.75)
    assert element.surface2.aperture.radius == pytest.approx(0.75)

    editor = _editor(window)
    assert editor["index"].value() == pytest.approx(1.5)
    assert editor["radius"].value() == pytest.approx(0.75)
    assert editor["s1"].text() == "-0.25"
    assert editor["s2"].text() == "0.25"
    window.close()


def test_ok_applies_to_new_element(qt_app, repo_root):
    window = OpticsMainWindow(make_display([], []))
    window.load_stack(repo_root / "tests" / "test_stack.json")
    _trigger_new_element(window)

    editor = _editor(window)
    editor["index"].setValue(1.9)
    _click(window, QtWidgets.QDialogButtonBox.StandardButton.Ok)

    assert window.elements[-1].material.refractive_index == pytest.approx(1.9)
    assert window.elements[0].material.refractive_index == pytest.approx(1.5)
    assert "Updated properties for lens3" in window.statusBar().currentMessage()
    window.close()


def _trigger_save_as(window):
    menu = window.findChild(QtWidgets.QMenu, "menuFile")
    assert menu is not None
    matches = [a for a in menu.actions() if a.objectName() == "actionSave_As"]
    assert matches, "actionSave_As missing from File menu"
    matches[0].trigger()


def test_save_as_action_writes_file(qt_app, repo_root, tmp_path, monkeypatch):
    window = OpticsMainWindow(make_display([], []))
    window.load_stack(repo_root / "tests" / "test_stack.json")

    out_path = tmp_path / "saved_stack.json"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(out_path), "")),
    )

    _trigger_save_as(window)

    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    expected = json.loads((repo_root / "tests" / "test_stack.json").read_text(encoding="utf-8"))
    assert saved == expected
    assert "Saved" in window.statusBar().currentMessage()
    window.close()


def test_save_as_appends_json_extension(qt_app, repo_root, tmp_path, monkeypatch):
    window = OpticsMainWindow(make_display([], []))
    window.load_stack(repo_root / "tests" / "test_stack.json")

    out_path = tmp_path / "stack_no_ext"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(out_path), "")),
    )

    _trigger_save_as(window)

    assert out_path.with_suffix(".json").exists()
    window.close()


def test_save_as_round_trip_preserves_edits(qt_app, repo_root, tmp_path, monkeypatch):
    window = OpticsMainWindow(make_display([], []))
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _editor(window)
    editor["index"].setValue(1.8)
    _click(window, QtWidgets.QDialogButtonBox.StandardButton.Ok)

    out_path = tmp_path / "modified_stack.json"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(out_path), "")),
    )
    _trigger_save_as(window)

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["lens1"]["material"]["refractive_index"] == pytest.approx(1.8)

    window2 = OpticsMainWindow(make_display([], []))
    elements = window2.load_stack(out_path)
    assert elements[0].material.refractive_index == pytest.approx(1.8)
    window.close()
    window2.close()


def test_save_as_empty_stack_warns(qt_app, tmp_path, monkeypatch):
    window = OpticsMainWindow(make_display([], []))

    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append(args)),
    )
    dialog_calls = []
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: dialog_calls.append(1) or ("", "")),
    )

    _trigger_save_as(window)

    assert warnings, "expected a warning dialog when no stack is loaded"
    assert not dialog_calls, "save dialog should not open when no stack is loaded"
    assert not list(tmp_path.iterdir())
    window.close()


def _source_editor(window):
    return {
        "type": window.findChild(QtWidgets.QComboBox, "sourceTypeCombo"),
        "power": window.findChild(QtWidgets.QDoubleSpinBox, "sourcePowerSpin"),
        "cone": window.findChild(QtWidgets.QDoubleSpinBox, "coneAngleSpin"),
        "rays": window.findChild(QtWidgets.QSpinBox, "numRaysSpin"),
        "x": window.findChild(QtWidgets.QDoubleSpinBox, "sourceX"),
        "y": window.findChild(QtWidgets.QDoubleSpinBox, "sourceY"),
        "z": window.findChild(QtWidgets.QDoubleSpinBox, "sourceZ"),
        "buttons": window.findChild(QtWidgets.QDialogButtonBox, "sourceButtonBox"),
    }


def _click_source(window, role):
    _source_editor(window)["buttons"].button(role).click()


def test_source_editor_enabled_without_stack(qt_app):
    window = OpticsMainWindow(make_display([], []))
    editor = _source_editor(window)
    assert editor["type"].isEnabled()
    assert editor["buttons"].isEnabled()
    window.close()


def test_source_panel_populated_with_defaults(qt_app):
    window = OpticsMainWindow(make_display([], []))
    editor = _source_editor(window)
    assert editor["type"].currentText() == "Single Point"
    assert editor["power"].value() == pytest.approx(1.0)
    assert editor["cone"].value() == pytest.approx(60.0)
    assert editor["rays"].value() == 16
    assert editor["z"].value() == pytest.approx(3.0)
    window.close()


def test_point_array_option_disabled(qt_app):
    window = OpticsMainWindow(make_display([], []))
    combo = _source_editor(window)["type"]
    item = combo.model().item(combo.findText("Point Array"))
    assert not item.isEnabled()
    window.close()


def test_source_sphere_drawn_at_startup(qt_app):
    canvas = make_display([], [])
    window = OpticsMainWindow(canvas)
    assert _mesh_count(canvas.view) == 1
    window.close()


def test_source_ok_applies_changes(qt_app):
    window = OpticsMainWindow(make_display([], []))
    editor = _source_editor(window)
    editor["rays"].setValue(8)
    editor["z"].setValue(5.0)
    editor["power"].setValue(2.5)
    _click_source(window, QtWidgets.QDialogButtonBox.StandardButton.Ok)

    assert window.source.num_rays == 8
    assert window.source.power == pytest.approx(2.5)
    assert window.source.position[2] == pytest.approx(5.0)
    assert len(window.rays) == 8
    assert all(ray.origin[2] == pytest.approx(5.0) for ray in window.rays)
    assert "Updated light source (8 rays)" in window.statusBar().currentMessage()
    window.close()


def test_source_cancel_reverts_edits(qt_app):
    window = OpticsMainWindow(make_display([], []))
    editor = _source_editor(window)
    original = editor["rays"].value()
    editor["rays"].setValue(original + 10)
    _click_source(window, QtWidgets.QDialogButtonBox.StandardButton.Cancel)

    assert editor["rays"].value() == original
    assert window.source.num_rays == original
    window.close()


def test_source_ok_retraces_rays(qt_app, repo_root):
    canvas = make_display([], [])
    window = OpticsMainWindow(canvas)
    window.load_stack(repo_root / "tests" / "test_stack.json")

    editor = _source_editor(window)
    editor["rays"].setValue(4)
    _click_source(window, QtWidgets.QDialogButtonBox.StandardButton.Ok)

    assert len(window.rays) == 4
    assert _mesh_count(canvas.view) >= 7
    window.close()


def test_source_survives_stack_reload(qt_app, repo_root):
    canvas = make_display([], [])
    window = OpticsMainWindow(canvas)
    editor = _source_editor(window)
    editor["rays"].setValue(6)
    _click_source(window, QtWidgets.QDialogButtonBox.StandardButton.Ok)

    window.load_stack(repo_root / "tests" / "test_stack.json")

    assert window.source.num_rays == 6
    assert len(window.rays) == 6
    assert _mesh_count(canvas.view) >= 7
    window.close()


def test_source_cone_angle_applies(qt_app):
    window = OpticsMainWindow(make_display([], []))
    editor = _source_editor(window)
    editor["cone"].setValue(90.0)
    _click_source(window, QtWidgets.QDialogButtonBox.StandardButton.Ok)

    assert window.source.cone_angle == pytest.approx(90.0)
    assert len(window.rays) == 16
    cos_half = np.cos(np.deg2rad(45.0))
    assert all(ray.direction[2] <= -cos_half + 1e-12 for ray in window.rays)
    window.close()


def test_source_cone_angle_cancel_reverts(qt_app):
    window = OpticsMainWindow(make_display([], []))
    editor = _source_editor(window)
    editor["cone"].setValue(45.0)
    _click_source(window, QtWidgets.QDialogButtonBox.StandardButton.Cancel)

    assert editor["cone"].value() == pytest.approx(60.0)
    assert window.source.cone_angle == pytest.approx(60.0)
    window.close()


def _trigger_measurement_plane(window):
    menu = window.findChild(QtWidgets.QMenu, "menuMeasure")
    assert menu is not None
    matches = [a for a in menu.actions() if a.objectName() == "actionMeasurement_Plane"]
    assert matches, "actionMeasurement_Plane missing from Measure menu"
    matches[0].trigger()


def test_measure_menu_has_action(qt_app):
    window = OpticsMainWindow(make_display([], []))
    menu_bar = window.findChild(QtWidgets.QMenuBar, "menubar")
    measure_menu = menu_bar.findChild(QtWidgets.QMenu, "menuMeasure")
    assert measure_menu is not None
    assert not measure_menu.isEmpty()
    assert any(a.objectName() == "actionMeasurement_Plane" for a in measure_menu.actions())
    window.close()


def test_measurement_plane_action_adds_plane(qt_app, repo_root):
    canvas = make_display([], [])
    window = OpticsMainWindow(canvas)
    window.load_stack(repo_root / "tests" / "test_stack.json")
    meshes_before = _mesh_count(canvas.view)

    _trigger_measurement_plane(window)

    assert len(window.planes) == 1
    assert _mesh_count(canvas.view) == meshes_before + 1
    assert "Added plane1" in window.statusBar().currentMessage()
    window.close()


def test_multiple_measurement_planes_offset(qt_app):
    window = OpticsMainWindow(make_display([], []))
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    assert len(window.planes) == 2
    assert window.planes[0].o[2] != window.planes[1].o[2]
    window.close()


def test_measurement_plane_persists_across_reload(qt_app, repo_root):
    canvas = make_display([], [])
    window = OpticsMainWindow(canvas)
    _trigger_measurement_plane(window)

    window.load_stack(repo_root / "tests" / "test_stack.json")

    assert len(window.planes) == 1
    assert _mesh_count(canvas.view) >= 7
    window.close()


def test_measurement_plane_remeasures_on_source_change(qt_app, repo_root):
    canvas = make_display([], [])
    window = OpticsMainWindow(canvas)
    window.load_stack(repo_root / "tests" / "test_stack.json")
    _trigger_measurement_plane(window)

    plane = window.planes[0]
    editor = _source_editor(window)
    editor["rays"].setValue(8)
    editor["cone"].setValue(30.0)
    _click_source(window, QtWidgets.QDialogButtonBox.StandardButton.Ok)

    assert len(window.planes) == 1
    assert plane.grid.shape == (64, 64)
    assert plane.grid.sum() >= 0.0
    window.close()


def _plane_editor(window):
    return {
        "selector": window.findChild(QtWidgets.QComboBox, "planeSelectorCombo"),
        "remove": window.findChild(QtWidgets.QPushButton, "removePlaneButton"),
        "pos": [
            window.findChild(QtWidgets.QDoubleSpinBox, "planePosXSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "planePosYSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "planePosZSpin"),
        ],
        "normal": [
            window.findChild(QtWidgets.QDoubleSpinBox, "planeNormalXSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "planeNormalYSpin"),
            window.findChild(QtWidgets.QDoubleSpinBox, "planeNormalZSpin"),
        ],
        "buttons": window.findChild(QtWidgets.QDialogButtonBox, "planeButtonBox"),
    }


def _click_plane(window, role):
    _plane_editor(window)["buttons"].button(role).click()


def test_plane_editor_disabled_without_planes(qt_app):
    window = OpticsMainWindow(make_display([], []))
    editor = _plane_editor(window)
    assert not editor["selector"].isEnabled()
    assert not editor["remove"].isEnabled()
    assert not editor["buttons"].isEnabled()
    window.close()


def test_add_plane_populates_selector_and_fields(qt_app):
    window = OpticsMainWindow(make_display([], []))
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    assert editor["selector"].isEnabled()
    assert [editor["selector"].itemText(i) for i in range(editor["selector"].count())] == ["plane1"]
    assert editor["pos"][2].value() == pytest.approx(-3.0)
    assert editor["normal"][2].value() == pytest.approx(1.0)
    window.close()


def test_plane_ok_applies_position(qt_app):
    window = OpticsMainWindow(make_display([], []))
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["pos"][2].setValue(-5.0)
    _click_plane(window, QtWidgets.QDialogButtonBox.StandardButton.Ok)

    assert window.planes[0].o[2] == pytest.approx(-5.0)
    assert "Updated properties for plane1" in window.statusBar().currentMessage()
    window.close()


def test_plane_ok_applies_normal(qt_app):
    window = OpticsMainWindow(make_display([], []))
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["normal"][2].setValue(0.5)
    _click_plane(window, QtWidgets.QDialogButtonBox.StandardButton.Ok)

    assert window.planes[0].n[2] == pytest.approx(1.0)
    window.close()


def test_plane_invalid_normal_warns_and_keeps(qt_app, monkeypatch):
    window = OpticsMainWindow(make_display([], []))
    _trigger_measurement_plane(window)

    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append(args)),
    )

    editor = _plane_editor(window)
    editor["normal"][0].setValue(1.0)
    editor["normal"][2].setValue(0.0)
    _click_plane(window, QtWidgets.QDialogButtonBox.StandardButton.Ok)

    assert warnings, "expected a warning for an invalid plane normal"
    np.testing.assert_allclose(window.planes[0].n, [0.0, 0.0, 1.0], atol=1e-9)
    assert "Updated properties" not in window.statusBar().currentMessage()
    window.close()


def test_plane_cancel_reverts_edits(qt_app):
    window = OpticsMainWindow(make_display([], []))
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    original = editor["pos"][2].value()
    editor["pos"][2].setValue(original - 2.0)
    _click_plane(window, QtWidgets.QDialogButtonBox.StandardButton.Cancel)

    assert editor["pos"][2].value() == pytest.approx(original)
    assert window.planes[0].o[2] == pytest.approx(original)
    window.close()


def test_plane_selector_switches_fields(qt_app):
    window = OpticsMainWindow(make_display([], []))
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    assert [editor["selector"].itemText(i) for i in range(editor["selector"].count())] == ["plane1", "plane2"]
    editor["selector"].setCurrentText("plane1")
    assert editor["pos"][2].value() == pytest.approx(-3.0)
    editor["selector"].setCurrentText("plane2")
    assert editor["pos"][2].value() == pytest.approx(-4.0)
    window.close()


def test_remove_plane_button(qt_app):
    window = OpticsMainWindow(make_display([], []))
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["remove"].click()

    assert len(window.planes) == 0
    assert editor["selector"].count() == 0
    assert not editor["selector"].isEnabled()
    assert "Removed plane1" in window.statusBar().currentMessage()
    window.close()


def test_remove_selected_plane_keeps_others(qt_app):
    window = OpticsMainWindow(make_display([], []))
    _trigger_measurement_plane(window)
    _trigger_measurement_plane(window)

    editor = _plane_editor(window)
    editor["selector"].setCurrentText("plane1")
    editor["remove"].click()

    assert len(window.planes) == 1
    assert [editor["selector"].itemText(i) for i in range(editor["selector"].count())] == ["plane2"]
    assert window.planes[0].o[2] == pytest.approx(-4.0)
    window.close()
