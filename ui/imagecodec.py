"""Qt-backed PNG codec for image-driven point-array sources.

The optics core stays Qt-free and dependency-free, so PNG (de)coding lives here
and is injected into :mod:`optics.source` via
:func:`optics.source.register_image_codec`. The grid convention matches the
source: a ``(H, W)`` float array in ``[0, 1]`` where brightness is per-point
emission weight. :func:`register` is called from the desktop app (and the source
editor, so headless tests that build a window still get a working codec).
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui

from optics.source import register_image_codec

__all__ = ["decode", "encode", "register"]

# Qt names the image format with a string at runtime; the stubs type it as
# ``bytes``. Annotate as Any so the runtime str passes type-checking.
_PNG_FORMAT: Any = "PNG"


def decode(data_b64: str) -> np.ndarray:
    """Decode a base64 PNG string to a ``(H, W)`` float grid in ``[0, 1]``."""
    try:
        blob = base64.b64decode(data_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Source image data is not valid base64") from exc
    image = QtGui.QImage.fromData(blob, _PNG_FORMAT)
    if image.isNull():
        raise ValueError("Could not decode source image (not a valid PNG)")
    rgba = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    width, height = rgba.width(), rgba.height()
    stride = rgba.bytesPerLine()
    scanlines = np.frombuffer(rgba.constBits(), dtype=np.uint8, count=stride * height).reshape(height, stride)
    rgb = scanlines[:, : width * 4].reshape(height, width, 4)[:, :, :3]
    # Match Qt's grayscale luma so encode/decode round-trips stay consistent.
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    return gray / 255.0


def encode(weights: np.ndarray) -> str:
    """Encode a ``(H, W)`` float grid in ``[0, 1]`` to a base64 PNG string."""
    grid = np.asarray(weights, dtype=float)
    if grid.ndim != 2:
        raise ValueError("Point array weights must be a 2D array")
    height, width = grid.shape
    pixels = np.clip(np.rint(grid * 255.0), 0, 255).astype(np.uint8)
    stride = (width + 3) // 4 * 4
    padded = np.zeros((height, stride), dtype=np.uint8)
    padded[:, :width] = pixels
    image = QtGui.QImage(
        padded.data, width, height, stride, QtGui.QImage.Format.Format_Grayscale8
    )
    block = QtCore.QByteArray()
    buffer = QtCore.QBuffer(block)
    buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    saved = image.save(buffer, _PNG_FORMAT)
    buffer.close()
    if not saved:
        raise ValueError("Failed to encode source image to PNG")
    return base64.b64encode(block.data()).decode("ascii")


def register() -> None:
    """Install this Qt codec into the Qt-free optics core."""
    register_image_codec(decode, encode)
