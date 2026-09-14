"""Irradiance source definition.

An :class:`IrradianceSource` carries a 2D irradiance pattern (power per unit
area) sampled on a rectangular grid. Tracing emits one ray per pixel, launched
from the pixel center in a configurable direction.
"""

from __future__ import annotations

import numpy as np


class IrradianceSource:
    def __init__(self, pattern, pitch=(1.0, 1.0), origin=(0.0, 0.0, 0.0),
                 direction=(0.0, 0.0, 1.0), name="source"):
        self.pattern = np.asarray(pattern, dtype=float)
        if self.pattern.ndim != 2:
            raise ValueError("irradiance pattern must be 2D")
        if np.isscalar(pitch):
            pitch = (float(pitch), float(pitch))
        self.pitch = (float(pitch[0]), float(pitch[1]))
        self.origin = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        nd = np.linalg.norm(d)
        self.direction = d / nd if nd > 0 else np.array([0.0, 0.0, 1.0])
        self.name = name

    @property
    def shape(self):
        return self.pattern.shape

    @property
    def area(self):
        """Total emitted power (sum of irradiance * pixel area)."""
        return float(np.sum(self.pattern) * self.pitch[0] * self.pitch[1])

    def pixel_centers(self):
        """Return ``(centers (H,W,3), uv (H,W,2))`` of pixel centers on the source plane."""
        H, W = self.pattern.shape
        pu, pv = self.pitch
        u = (np.arange(W) + 0.5) * pu - (W * pu) / 2.0
        v = (np.arange(H) + 0.5) * pv - (H * pv) / 2.0
        U, V = np.meshgrid(u, v)
        centers = np.stack([U + self.origin[0], V + self.origin[1],
                            np.full((H, W), self.origin[2])], axis=-1)
        return centers, np.stack([U, V], axis=-1)

    def flatten(self):
        """Return ``(origins (N,3), dirs (N,3), uv (N,2), irradiance (N,))`` for tracing."""
        centers, uv = self.pixel_centers()
        H, W = self.pattern.shape
        origins = centers.reshape(-1, 3)
        dirs = np.tile(self.direction, (H * W, 1))
        irr = self.pattern.reshape(-1)
        return origins, dirs, uv.reshape(-1, 2), irr

    def __repr__(self):
        return f"IrradianceSource(name={self.name!r}, shape={self.shape}, pitch={self.pitch})"
