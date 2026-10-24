"""Tests for the Zemax .zmx importer (parse -> elements -> stack -> trace)."""

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from optics.config import stack_from_config
from optics.expression import compile_parametric_surface
from optics.importers.zmx import (
    UnsupportedZmxError,
    conic_sag_expression,
    load_zmx_elements,
    parse_zmx,
    zmx_to_elements,
)
from optics.io import load_project
from optics.measurement_plane import MeasurementPlane
from optics.source import PointSource
from optics.system import OpticalSystem

ZMX_DIR = Path(__file__).parent / "zmx"


def _load_quiet_elements(path: Path) -> tuple[dict[str, Any], Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_zmx_elements(path)


def _modern_case(
    tmp_path: Path,
    surf_lines: list[str],
    last: str = 'SURF 9 "IMA Standard" IMA 0 0 NULL 12.5 0 0 0 0 0',
) -> Path:
    lines = [
        'FILE VER 1602 1 "case"',
        "UNIT MF MM",
        'SURF 0 "OBJ Standard" OBJ 0 25 NULL 12.5 0 0 0 0 0',
        *surf_lines,
        last,
        'GLAS "N-BK7" 1.5168 64.17 "N-BK7"',
        "END",
    ]
    path = tmp_path / "case.zmx"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# -- parsing -----------------------------------------------------------------
def test_parse_modern_dialect() -> None:
    prescription = parse_zmx(ZMX_DIR / "biconvex.zmx")
    assert [s.index for s in prescription.surfaces] == [0, 1, 2, 3]
    assert prescription.units == "MM"
    assert prescription.wavelengths == pytest.approx([0.58756, 0.486133, 0.656275])
    front = prescription.surfaces[1]
    assert front.surface_type == "STANDARD"
    assert front.curvature == pytest.approx(0.02)
    assert front.thickness == pytest.approx(5.0)
    assert front.glass == "N-BK7"
    assert front.semi_diameter == pytest.approx(12.5)
    assert front.vertex_z == pytest.approx(25.0)
    assert prescription.surfaces[2].vertex_z == pytest.approx(30.0)
    assert prescription.glasses["N-BK7"][0] == pytest.approx(1.51679364)


def test_parse_legacy_dialect() -> None:
    prescription = parse_zmx(ZMX_DIR / "cemented_doublet.zmx")
    assert len(prescription.surfaces) == 5
    s1, s2, s3 = prescription.surfaces[1:4]
    assert s1.surface_type == "STANDARD"
    assert s1.glass == "N-BK7"
    assert s1.curvature == pytest.approx(0.02)
    assert s1.semi_diameter == pytest.approx(10.0)
    assert s2.glass == "N-F2"
    assert s3.glass == ""
    assert s1.vertex_z == pytest.approx(20.0)
    assert s2.vertex_z == pytest.approx(24.0)
    assert s3.vertex_z == pytest.approx(27.0)


# -- element conversion ------------------------------------------------------
def test_element_geometry() -> None:
    elements, geometry = _load_quiet_elements(ZMX_DIR / "biconvex.zmx")
    assert list(elements) == ["lens1"]
    lens = elements["lens1"]
    assert lens["position"][2] == pytest.approx(-25.0)
    assert lens["material"]["refractive_index"] == pytest.approx(1.51679364)
    assert lens["surface1"]["aperture"]["radius"] == pytest.approx(12.5)
    assert lens["surface1"]["u_range"][0] == pytest.approx(-15.625)
    # The stack spans one Zemax thickness in mirrored simulator z.
    assert geometry.front_z == pytest.approx(-25.0)
    assert geometry.last_z == pytest.approx(-30.0)
    assert geometry.length == pytest.approx(5.0)
    assert geometry.radius == pytest.approx(12.5)


def test_elements_only_has_no_project() -> None:
    elements, _ = _load_quiet_elements(ZMX_DIR / "biconvex.zmx")
    assert list(elements) == ["lens1"]
    # An elements-only import carries no synthesized source/planes/version keys.
    for stray in ("source", "measurement_planes", "version"):
        assert stray not in elements
        assert stray not in elements["lens1"]


def test_load_zmx_elements_emits_warnings() -> None:
    with pytest.warns(UserWarning, match="dispersion"):
        load_zmx_elements(ZMX_DIR / "biconvex.zmx")


def test_curvature_is_mirrored_into_simulator_z() -> None:
    elements, _ = _load_quiet_elements(ZMX_DIR / "biconvex.zmx")
    element = stack_from_config(elements)[0]
    assert element.surface1.evaluate_local(0.0, 0.0)[2] == pytest.approx(0.0)
    # Front surface bulges toward the source (+Z): the rim sag is below the vertex.
    assert element.surface1.evaluate_local(1.0, 0.0)[2] < 0.0
    # Rear vertex sits one Zemax thickness down in simulator z, bulging toward -Z.
    assert element.surface2.evaluate_local(0.0, 0.0)[2] == pytest.approx(-5.0)
    assert element.surface2.evaluate_local(1.0, 0.0)[2] > -5.0


def test_plano_convex_flat_front_and_rear_offset() -> None:
    elements, _ = _load_quiet_elements(ZMX_DIR / "plano_convex.zmx")
    lens = elements["lens1"]
    assert lens["surface1"]["z"] == "0"
    assert lens["surface2"]["z"].startswith("-4.0 + ")


def test_doublet_pairing_and_materials() -> None:
    elements, _ = _load_quiet_elements(ZMX_DIR / "doublet.zmx")
    assert list(elements) == ["lens1", "lens2"]
    lens1, lens2 = elements.values()
    assert lens1["position"][2] == pytest.approx(-25.0)
    assert lens2["position"][2] == pytest.approx(-34.0)
    assert lens1["material"]["refractive_index"] == pytest.approx(1.5168)
    # N-F2 has no GLAS entry in the fixture: it must come from the built-in table.
    assert lens2["material"]["refractive_index"] == pytest.approx(1.64997)
    assert lens2["surface2"]["z"].startswith("-2.0 + ")


def test_cemented_doublet_shares_the_interface() -> None:
    elements, _ = _load_quiet_elements(ZMX_DIR / "cemented_doublet.zmx")
    stack = stack_from_config(elements)
    assert len(stack) == 2
    first, second = stack
    shared_front = second.surface1.point(0.0, 0.0)
    shared_rear = first.surface2.point(0.0, 0.0)
    assert np.allclose(shared_rear, shared_front, atol=1e-12)
    assert shared_rear[2] == pytest.approx(-24.0)
    assert first.transform.translation[2] == pytest.approx(-20.0)
    assert second.material.refractive_index == pytest.approx(1.64997)


# -- warnings ------------------------------------------------------------------
def test_monochromatic_import_warning() -> None:
    with pytest.warns(UserWarning, match="dispersion"):
        load_zmx_elements(ZMX_DIR / "biconvex.zmx")


def test_builtin_table_warning() -> None:
    with pytest.warns(UserWarning, match="built-in table"):
        load_zmx_elements(ZMX_DIR / "doublet.zmx")


# -- rejections ----------------------------------------------------------------
def test_reject_mirror(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path, ['SURF 1 "STANDARD" STANDARD 0.02 5 MIRROR 12.5 0 0 0 0 0']
    )
    with pytest.raises(UnsupportedZmxError, match="[Mm]irror"):
        load_zmx_elements(path)


def test_reject_aspheric_coefficients(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path,
        [
            'SURF 1 "STANDARD" STANDARD 0.02 5 N-BK7 12.5 0 1e-4',
            'SURF 2 "STANDARD" STANDARD -0.02 10 NULL 12.5 0',
        ],
    )
    with pytest.raises(UnsupportedZmxError, match="[Aa]spheric"):
        load_zmx_elements(path)


def test_reject_unsupported_surface_type(tmp_path: Path) -> None:
    path = _modern_case(tmp_path, ['SURF 1 "ASP" EVENASPHERE 0.02 5 N-BK7 12.5 0'])
    with pytest.raises(UnsupportedZmxError, match="EVENASPHERE"):
        load_zmx_elements(path)


def test_reject_unknown_glass(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path,
        [
            'SURF 1 "STANDARD" STANDARD 0.02 5 ZGL-1 12.5 0',
            'SURF 2 "STANDARD" STANDARD -0.02 10 NULL 12.5 0',
        ],
    )
    with pytest.raises(UnsupportedZmxError, match="Unknown glass"):
        load_zmx_elements(path)


def test_reject_missing_semi_diameter(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path,
        [
            'SURF 1 "STANDARD" STANDARD 0.02 5 N-BK7 0 0',
            'SURF 2 "STANDARD" STANDARD -0.02 10 NULL 12.5 0',
        ],
    )
    with pytest.raises(UnsupportedZmxError, match="semi-diameter"):
        load_zmx_elements(path)


def test_reject_glass_after_final_surface(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path,
        ['SURF 1 "STANDARD" STANDARD 0.02 5 N-BK7 12.5 0'],
        last='SURF 9 "STANDARD" STANDARD -0.01 0 N-F2 12.5 0',
    )
    with pytest.raises(UnsupportedZmxError, match="exit surface"):
        load_zmx_elements(path)


def test_reject_no_glass(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path, ['SURF 1 "STANDARD" STANDARD 0.02 5 NULL 12.5 0']
    )
    with pytest.raises(UnsupportedZmxError, match="[Nn]o refracting glass"):
        load_zmx_elements(path)


def test_curved_dummy_rejected(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path,
        [
            'SURF 1 "STANDARD" STANDARD 0.02 2 N-BK7 12.5 0',
            'SURF 2 "STANDARD" STANDARD 0.01 1 N-BK7 12.5 0',
            'SURF 3 "STANDARD" STANDARD -0.02 10 NULL 12.5 0',
        ],
    )
    with pytest.raises(UnsupportedZmxError, match="single glass"):
        load_zmx_elements(path)


def test_dummy_surface_merged(tmp_path: Path) -> None:
    path = _modern_case(
        tmp_path,
        [
            'SURF 1 "STANDARD" STANDARD 0.02 2 N-BK7 12.5 0',
            'SURF 2 "STANDARD" STANDARD 0 1 N-BK7 12.5 0',
            'SURF 3 "STANDARD" STANDARD -0.02 10 NULL 12.5 0',
        ],
    )
    prescription = parse_zmx(path)
    elements, _ = zmx_to_elements(prescription)
    assert any("Dummy surface" in message for message in prescription.warnings)
    lens = elements["lens1"]
    assert lens["position"][2] == pytest.approx(-25.0)
    assert lens["surface2"]["z"].startswith("-3.0 + ")


# -- expression generation ------------------------------------------------------
def test_conic_flat_is_zero() -> None:
    assert conic_sag_expression(0.0, 0.0) == "0"
    assert conic_sag_expression(0.0, -1.0) == "0"


def test_conic_expression_evaluates_natively() -> None:
    curvature, conic = 0.02, -0.5
    expression = conic_sag_expression(-curvature, conic)
    compiled = compile_parametric_surface(
        "u", "v", expression, (-10.0, 10.0), (-10.0, 10.0)
    )
    u, v = 0.5, 0.3
    r2 = u * u + v * v
    expected = (-curvature * r2) / (1 + math.sqrt(1 - (1 + conic) * curvature**2 * r2))
    assert compiled.eval_local(u, v)[2] == pytest.approx(expected)


# -- end-to-end trace smoke ------------------------------------------------------
def _trace_total(elements: dict[str, Any], geometry: Any) -> float:
    """Trace the imported stack against a beam sized from ``geometry``."""
    system = OpticalSystem(stack_from_config(elements))
    margin_front = max(2.0 * geometry.length, 4.0 * geometry.radius, 1.0)
    margin_back = max(geometry.length, 2.0 * geometry.radius, 1.0)
    half = max(2.0 * geometry.radius, 1.0)
    source = PointSource(
        position=[0.0, 0.0, geometry.front_z + margin_front],
        power=1.0,
        num_rays=256,
        source_type="single_point",
        cone_angle=5.0,
    )
    plane = MeasurementPlane(
        origin=np.array([0.0, 0.0, geometry.last_z - margin_back]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=[-half, half],
        v_range=[-half, half],
        resolution=[64, 64],
    )
    origins, dirs, powers = source.emit_arrays()
    result = MeasurementPlane.measure_batch_arrays(
        [plane], origins, dirs, powers, system
    )[0]
    return result.total


def test_trace_smoke_biconvex() -> None:
    elements, geometry = _load_quiet_elements(ZMX_DIR / "biconvex.zmx")
    assert _trace_total(elements, geometry) > 0.5


def test_trace_smoke_cemented_zero_gap_interface() -> None:
    elements, geometry = _load_quiet_elements(ZMX_DIR / "cemented_doublet.zmx")
    assert _trace_total(elements, geometry) > 0.5


# -- optics.io: JSON-only loading ------------------------------------------------
def test_load_project_rejects_zmx() -> None:
    with pytest.raises(ValueError, match="Unsupported project file extension"):
        load_project(ZMX_DIR / "biconvex.zmx")


def test_load_project_json(tmp_path: Path) -> None:
    path = tmp_path / "proj.json"
    path.write_text(json.dumps({"elements": {}}), encoding="utf-8")
    assert load_project(path) == {"elements": {}}


def test_load_project_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_project(path)


def test_load_project_rejects_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "proj.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported project file extension"):
        load_project(path)
