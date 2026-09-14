"""Shared pytest fixtures.

The ``optics`` core runs on the CPU (no GL). The ``gui`` package needs a Qt
platform; we force the offscreen platform and a core-profile surface format so
tests run on headless CI without a display or GPU.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _qt_offscreen():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QSurfaceFormat

    fmt = QSurfaceFormat()
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setMajorVersion(3)
    fmt.setMinorVersion(3)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    if not QApplication.instance():
        app = QApplication.instance() or QApplication(sys.argv)
    yield
    if QApplication.instance() is not None:
        QApplication.instance().quit()


@pytest.fixture
def gaussian_pattern(grid=80, sigma_frac=0.2, peak=1.0):
    a = np.arange(grid) - grid / 2.0
    env = peak * np.exp(-(a ** 2) / (2 * (grid * sigma_frac) ** 2))
    return (env[None, :] * env[:, None]).astype(np.float32)
