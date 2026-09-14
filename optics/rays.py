"""Ray stack: a batch of rays traced through an optical stack.

The single tracing backend is the Numba kernel in :mod:`optics.accel` (Numba is
a required dependency).
"""

from __future__ import annotations

import numpy as np

from . import accel


class RayStack:
    def __init__(self, origins, dirs):
        self.origins = np.asarray(origins, dtype=float)
        self.dirs = np.asarray(dirs, dtype=float)
        if self.origins.shape != self.dirs.shape:
            raise ValueError("origins and dirs must share shape")
        if self.origins.ndim != 2 or self.origins.shape[1] != 3:
            raise ValueError("origins/dirs must have shape (N, 3)")
        self._normalize()

    def _normalize(self):
        n = np.sqrt(np.sum(self.dirs * self.dirs, axis=-1, keepdims=True))
        self.dirs = self.dirs / np.maximum(n, 1e-30)

    # --- propagation ------------------------------------------------------
    def propagate(self, distance):
        """Advance every ray by ``distance`` (scalar or (N,) array)."""
        distance = np.asarray(distance)
        if distance.ndim == 0:
            distance = np.full(self.origins.shape[0], float(distance))
        self.origins = self.origins + self.dirs * distance[:, None]
        return self

    def propagate_to_plane(self, z):
        """Propagate rays until they reach the plane ``z`` (returns hit points)."""
        t = (z - self.origins[:, 2]) / self.dirs[:, 2]
        self.origins = self.origins + t[:, None] * self.dirs
        return self.origins.copy()

    # --- tracing ----------------------------------------------------------
    def trace(self, surfaces, media):
        """Trace through ``surfaces`` with medium indices ``media`` (len S+1).

        Returns a :class:`optics.stack.TraceResult`.
        """
        from .stack import TraceResult

        origins, dirs = self.origins, self.dirs
        original_origins = origins.copy()
        hits, tir, blocked = accel.trace_numba(origins, dirs, surfaces, media)
        result = TraceResult(surfaces, hits, tir, dirs, blocked)
        result.source_points = original_origins
        return result
