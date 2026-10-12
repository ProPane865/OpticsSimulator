"""Versioned project document format: elements, light source, measurement planes."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .measurement_plane import MeasurementPlane
from .source import POINT_ARRAY, SINGLE_POINT, PointSource

PROJECT_VERSION = 1

DEFAULT_SOURCE_CONFIG: dict[str, Any] = {
    "position": [0.0, 0.0, 3.0],
    "power": 1.0,
    "num_rays": 16,
    "source_type": SINGLE_POINT,
    "cone_angle": 60.0,
}

DEFAULT_MEASUREMENT_CONFIG: dict[str, Any] = {
    "origin": [0.0, 0.0, -3.0],
    "normal": [0.0, 0.0, 1.0],
    "u_range": [-2.0, 2.0],
    "v_range": [-2.0, 2.0],
    "resolution": [64, 64],
}


def source_from_config(config: object) -> PointSource:
    if not isinstance(config, dict):
        raise ValueError("Source config must be an object")
    source_type = str(config.get("source_type", SINGLE_POINT))
    if source_type == POINT_ARRAY:
        raise ValueError("Point array sources are not supported yet")
    return PointSource(
        position=np.asarray(config.get("position", [0.0, 0.0, 3.0]), dtype=float),
        power=float(config.get("power", 1.0)),
        num_rays=int(config.get("num_rays", 16)),
        source_type=source_type,
        cone_angle=float(config.get("cone_angle", 60.0)),
    )


def source_to_config(source: PointSource) -> dict[str, Any]:
    return {
        "position": [float(v) for v in source.position],
        "power": float(source.power),
        "num_rays": int(source.num_rays),
        "source_type": str(source.source_type),
        "cone_angle": float(source.cone_angle),
    }


def plane_from_config(config: object) -> MeasurementPlane:
    if not isinstance(config, dict):
        raise ValueError("Measurement plane config must be an object")
    for key in ("origin", "normal", "u_range", "v_range", "resolution"):
        if key not in config:
            raise ValueError(f"Missing required key {key!r} in measurement plane config")
    return MeasurementPlane(
        origin=np.asarray(config["origin"], dtype=float),
        normal=np.asarray(config["normal"], dtype=float),
        u_range=config["u_range"],
        v_range=config["v_range"],
        resolution=config["resolution"],
    )


def plane_to_config(name: str, plane: MeasurementPlane) -> dict[str, Any]:
    return {
        "name": name,
        "origin": [float(v) for v in plane.o],
        "normal": [float(v) for v in plane.n],
        "u_range": [plane.u_min, plane.u_max],
        "v_range": [plane.v_min, plane.v_max],
        "resolution": [plane.ru, plane.rv],
    }


def parse_project(
    document: object,
) -> "tuple[dict[str, Any], PointSource | None, list[tuple[str, MeasurementPlane]] | None]":
    """Split a project document into ``(elements_config, source, planes)``.

    Files without an ``"elements"`` key are treated as legacy element-only
    maps; ``source``/``planes`` are then ``None`` meaning "keep current".
    """
    if not isinstance(document, dict):
        raise ValueError("Optical stack must be a JSON object")
    if "elements" in document:
        elements_config = document["elements"]
        source_config = document.get("source")
        planes_config = document.get("measurement_planes")
    else:
        elements_config = document
        source_config = None
        planes_config = None

    source = None if source_config is None else source_from_config(source_config)

    planes: list[tuple[str, MeasurementPlane]] | None = None
    if planes_config is not None:
        if not isinstance(planes_config, list):
            raise ValueError("measurement_planes must be a list")
        planes = []
        for entry in planes_config:
            base_name = str(entry.get("name", "")) if isinstance(entry, dict) else ""
            planes.append((base_name, plane_from_config(entry)))

    return elements_config, source, planes


def build_project_document(
    elements_schema: Mapping[str, Any],
    source: PointSource,
    plane_names: Sequence[str],
    planes: Sequence[MeasurementPlane],
) -> dict[str, Any]:
    """Assemble the versioned project document for saving."""
    return {
        "version": PROJECT_VERSION,
        "elements": dict(elements_schema),
        "source": source_to_config(source),
        "measurement_planes": [
            plane_to_config(name, plane)
            for name, plane in zip(plane_names, planes)
        ],
    }
