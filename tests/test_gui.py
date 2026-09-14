"""GUI layer tests (offscreen): Scene model, source sync, viewport refresh."""

import numpy as np
import pytest

from gui.models import Scene, gaussian_pattern, disk_pattern


@pytest.fixture
def scene():
    return Scene("bi_convex")


def test_preset_builds_two_surfaces(scene):
    assert len(scene.surfaces) == 2
    assert scene.surfaces[0]["kind"] == "conic"
    assert scene.detector_z > 0


def test_all_presets_construct():
    for name in ("bi_convex", "thin_lens", "ball_lens", "collimator"):
        s = Scene(name)
        tr, irr = s.trace()
        assert tr.n_rays > 0
        assert irr["irradiance"].size == tr.grid_shape[0] * tr.grid_shape[1]


def test_trace_returns_trace_and_irradiance(scene):
    tr, irr = scene.trace()
    assert hasattr(tr, "hits")
    assert "irradiance" in irr and "valid" in irr


def test_source_pattern_reflected_in_scene(scene):
    assert scene.pattern.shape == (scene.grid, scene.grid)
    assert scene.source.pattern.shape == (scene.grid, scene.grid)


def test_gaussian_pattern_peak_and_shape():
    # The 2D pattern is an outer product, so its peak is peak**2.
    pat = gaussian_pattern(64, sigma_frac=0.2, peak=2.5)
    assert pat.shape == (64, 64)
    assert pat.max() == pytest.approx(2.5 ** 2)


def test_disk_pattern_peak_and_shape():
    pat = disk_pattern(64, peak=3.0)
    assert pat.shape == (64, 64)
    assert pat.max() == pytest.approx(3.0)
    assert pat.min() == 0.0


def test_spot_size_returns_fz_spot_n(scene):
    fz, spot, n = scene.spot_size()
    assert n == scene.grid ** 2  # number of rays
    assert spot > 0.0
    assert fz != 0.0


def test_add_and_remove_surface(scene):
    n0 = len(scene.surfaces)
    scene.add_plane(n=1.0, z0=10.0)
    assert len(scene.surfaces) == n0 + 1
    scene.remove_surface(n0)  # remove the just-added surface by index
    assert len(scene.surfaces) == n0


def test_rebuild_source_updates_pattern(scene):
    scene.pattern_peak = 0.0
    scene._rebuild_source()
    assert scene.source.pattern.max() == 0.0
    scene.pattern_peak = 1.0
    scene._rebuild_source()
    assert scene.source.pattern.max() == pytest.approx(1.0)


def test_disk_pattern_source_traces(scene):
    scene.pattern_type = "disk"
    scene.pattern_peak = 1.0
    scene._rebuild_source()
    tr, irr = scene.trace()
    assert irr["irradiance"].max() > 0.0


# --- SourcePanel sync (the zero-irradiance regression) --------------------

def test_source_panel_sync_preserves_peak():
    from gui.panels import SourcePanel
    from gui.viewport import Viewport

    scene = Scene("bi_convex")
    vp = Viewport()
    panel = SourcePanel(vp, scene)
    # After construction the source pattern must reflect the scene peak (1.0),
    # not the default spinbox value (0.0).
    assert scene.pattern_peak == pytest.approx(1.0)
    assert scene.source.pattern.max() == pytest.approx(1.0)


def test_source_panel_ui_edit_propagates_to_scene():
    from gui.panels import SourcePanel
    from gui.viewport import Viewport
    from PySide6.QtWidgets import QApplication

    scene = Scene("bi_convex")
    vp = Viewport()
    panel = SourcePanel(vp, scene)
    app = QApplication.instance()
    app.processEvents()

    panel.peak.setValue(4.0)
    app.processEvents()
    assert scene.pattern_peak == pytest.approx(4.0)
    # 2D pattern peak is peak**2 (outer product of the 1D envelope).
    assert scene.source.pattern.max() == pytest.approx(16.0)


def test_source_panel_type_switch_keeps_peak():
    from gui.panels import SourcePanel
    from gui.viewport import Viewport
    from PySide6.QtWidgets import QApplication

    scene = Scene("bi_convex")
    vp = Viewport()
    panel = SourcePanel(vp, scene)
    app = QApplication.instance()
    app.processEvents()

    panel.pattern_type.setCurrentText("disk")
    app.processEvents()
    # Peak must survive a pattern-type change (previously reset to 0).
    assert scene.pattern_peak == pytest.approx(1.0)
    assert scene.source.pattern.max() == pytest.approx(1.0)
    assert scene.pattern_type == "disk"


def test_main_window_trace_nonzero_irradiance():
    from gui.app import MainWindow
    from PySide6.QtWidgets import QApplication

    win = MainWindow("bi_convex")
    scene = win.scene
    tr, irr = scene.trace()
    peak = scene.source.pattern.max()
    assert peak > 0.0
    assert irr["irradiance"].max() > 0.0
    assert irr["valid"].mean() > 0.3


def test_main_window_switch_preset_keeps_peak():
    from gui.app import MainWindow
    from PySide6.QtWidgets import QApplication

    win = MainWindow("bi_convex")
    win._switch("ball_lens")
    assert win.scene.pattern_peak == pytest.approx(1.0)
    assert win.scene.source.pattern.max() == pytest.approx(1.0)


def test_viewport_refresh_without_crash():
    from gui.app import MainWindow

    win = MainWindow("thin_lens")
    win.viewport.refresh()  # builds VisPy visuals under a live (offscreen) context
    assert win.viewport is not None


@pytest.mark.parametrize("preset", ["bi_convex", "thin_lens", "ball_lens", "collimator"])
def test_viewport_ray_geometry_builds(preset):
    # Each ray yields n_surfaces+1 segments; blocked rays (per-ray) must map
    # onto per-segment colors without a shape mismatch.
    from gui.viewport import Viewport

    tr = _trace(preset)
    segs_per_ray = tr.n_surfaces + 1
    n_segs = tr.n_rays * segs_per_ray
    cols = Viewport._ray_segment_colors(segs_per_ray, tr.blocked, n_segs)
    assert cols.shape == (n_segs, 4)

    orange = np.array([0.95, 0.7, 0.2, 0.85], np.float32)
    gray = np.array([0.5, 0.5, 0.55, 0.85], np.float32)
    seg_blocked = np.repeat(tr.blocked, segs_per_ray)
    # Every segment of a blocked ray is gray; at least one orange ray remains.
    assert np.all(np.all(cols[seg_blocked] == gray, axis=1))
    assert np.any(np.all(cols[~seg_blocked] == orange, axis=1))


def _trace(preset):
    from gui.app import MainWindow

    return MainWindow(preset).scene.trace()[0]
