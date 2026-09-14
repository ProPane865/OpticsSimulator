"""Optical stack: an ordered list of surfaces separated by media, plus tracing."""

from __future__ import annotations

import numpy as np

from .rays import RayStack


class TraceResult:
    """Container for the result of tracing a source through a stack."""

    def __init__(self, surfaces, hits, tir, final_dirs, blocked=None):
        self.surfaces = surfaces
        self.hits = hits            # (S, N, 3) hit point on each surface
        self.tir = tir              # (S, N) total-internal-reflection flags
        self.final_dirs = final_dirs  # (N, 3) ray directions after last surface
        self.blocked = None if blocked is None else blocked.astype(bool)
        self.irradiance_input = None  # (N,) input irradiance per ray
        self.uv = None              # (N, 2) input-plane coordinates per ray
        self.source_points = None   # (N, 3) ray origins (source plane)
        self.grid_shape = None      # (H, W) source grid shape
        self.pitch = (1.0, 1.0)     # (pu, pv) source pixel pitch

    @property
    def n_surfaces(self):
        return len(self.surfaces)

    @property
    def n_rays(self):
        return self.hits.shape[1]

    def hits_on(self, index):
        return self.hits[index]

    def propagate_to(self, z):
        """Propagate final rays to plane ``z``; return ``(points (N,3), t (N,))``."""
        d = self.final_dirs
        t = (z - self.hits[-1][:, 2]) / d[:, 2]
        points = self.hits[-1] + t[:, None] * d
        return points, t

    def __repr__(self):
        return f"TraceResult(n_rays={self.n_rays}, n_surfaces={self.n_surfaces})"


class OpticalStack:
    def __init__(self, name="stack"):
        self.name = name
        self.surfaces = []
        self.media = [1.0]  # medium index before the first surface

    def add_surface(self, surface, medium_index=1.0):
        """Append a surface; ``medium_index`` is the index after it."""
        self.surfaces.append(surface)
        self.media.append(float(medium_index))
        return surface

    def add_media(self, index):
        self.media.append(float(index))

    def trace(self, source):
        origins, dirs, uv, irr = source.flatten()
        original_origins = origins.copy()
        stack = RayStack(origins, dirs)
        result = stack.trace(self.surfaces, self.media)
        result.irradiance_input = irr
        result.uv = uv
        result.source_points = original_origins
        result.grid_shape = source.shape
        result.pitch = source.pitch
        return result

    def __repr__(self):
        return f"OpticalStack(name={self.name!r}, surfaces={[s.name for s in self.surfaces]})"
