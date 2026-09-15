"""Scene model: source + optical stack + detector, with presets."""

from __future__ import annotations

import numpy as np

from optics import OpticalStack, ConicSurface, Plane, IrradianceSource
from optics import irradiance_from_trace


def gaussian_pattern(grid, sigma_frac=0.2, peak=1.0):
    a = (np.arange(grid) - grid / 2)
    env = peak * np.exp(-(a ** 2) / (2 * (grid * sigma_frac) ** 2))
    return (env[None, :] * env[:, None]).astype(np.float32)


def disk_pattern(grid, radius_frac=0.9, peak=1.0):
    a = (np.arange(grid) - grid / 2)
    X, Y = np.meshgrid(a, a)
    r = np.sqrt(X * X + Y * Y) / (grid / 2)
    pat = np.where(r < radius_frac, peak, 0.0)
    return pat.astype(np.float32)


class Scene:
    """Holds the full simulation state and produces trace results."""

    def __init__(self, preset="bi_convex"):
        self.grid = 96
        self.pitch = 1.0
        self.pattern_type = "gaussian"
        self.pattern_sigma = 0.2
        self.pattern_peak = 1.0
        self.source_origin = (0.0, 0.0, -20.0)
        self.source_direction = (0.0, 0.0, 1.0)
        self.detector_z = 60.0
        self.surfaces = []
        self._build(preset)

    # --- construction -----------------------------------------------------
    def _build(self, preset):
        self.surfaces = []
        self.detector_z = 60.0
        presets = {
            "bi_convex": self._bi_convex,
            "thin_lens": self._thin_lens,
            "ball_lens": self._ball_lens,
            "collimator": self._collimator,
        }
        presets.get(preset, self._bi_convex)()
        self._rebuild_source()

    def _bi_convex(self):
        R = 50.0
        self.add_conic(radius=R, conicity=0.0, z0=0.0, n=1.5, aperture=45.0)
        self.add_conic(radius=-R, conicity=0.0, z0=6.0, n=1.0, aperture=45.0)
        self.detector_z = 70.0

    def _thin_lens(self):
        R = 60.0
        self.add_conic(radius=R, conicity=0.0, z0=0.0, n=1.5, aperture=50.0)
        self.add_plane(n=1.0, z0=2.0)
        self.detector_z = 85.0

    def _ball_lens(self):
        R = 30.0
        self.add_sphere(radius=R, z0=0.0, n=1.5, aperture=R)
        self.add_plane(n=1.0, z0=0.0)
        self.detector_z = 60.0

    def _collimator(self):
        # Reverse bi-convex: would focus a point source -> collimate.
        R = 50.0
        self.add_conic(radius=-R, conicity=0.0, z0=6.0, n=1.5, aperture=45.0)
        self.add_conic(radius=R, conicity=0.0, z0=0.0, n=1.0, aperture=45.0)
        self.source_direction = (0.0, 0.0, -1.0)
        self.source_origin = (0.0, 0.0, 20.0)
        self.detector_z = -40.0

    # --- surface editing --------------------------------------------------
    def add_conic(self, radius, conicity=0.0, z0=0.0, n=1.0, aperture=None):
        self.surfaces.append({
            "kind": "conic", "name": f"conic{len(self.surfaces)}",
            "radius": float(radius), "conicity": float(conicity),
            "z0": float(z0), "n": float(n), "aperture": aperture,
        })
        return self

    def add_sphere(self, radius, z0=0.0, n=1.0, aperture=None):
        return self.add_conic(radius=radius, conicity=0.0, z0=z0, n=n, aperture=aperture)

    def add_plane(self, n=1.0, z0=0.0, aperture=None):
        self.surfaces.append({
            "kind": "plane", "name": f"plane{len(self.surfaces)}",
            "z0": float(z0), "n": float(n), "aperture": aperture,
        })
        return self

    def remove_surface(self, index):
        if 0 <= index < len(self.surfaces):
            del self.surfaces[index]

    def _rebuild_source(self):
        if self.pattern_type == "gaussian":
            pat = gaussian_pattern(self.grid, self.pattern_sigma, self.pattern_peak)
        elif self.pattern_type == "disk":
            pat = disk_pattern(self.grid, peak=self.pattern_peak)
        else:
            pat = np.ones((self.grid, self.grid), dtype=np.float32) * self.pattern_peak
        self.pattern = pat
        self.source = IrradianceSource(
            pat, pitch=self.pitch, origin=self.source_origin,
            direction=self.source_direction, name="source",
        )

    # --- tracing ----------------------------------------------------------
    def build_stack(self):
        stack = OpticalStack("scene")
        for s in self.surfaces:
            if s["kind"] == "conic":
                surf = ConicSurface(
                    s["name"], radius=s["radius"], conicity=s["conicity"],
                    z0=s["z0"], aperture=s["aperture"],
                )
            else:
                surf = Plane(s["name"], z0=s["z0"], aperture=s["aperture"])
            stack.add_surface(surf, medium_index=s["n"])
        if stack.media[-1] != 1.0 and self.surfaces:
            stack.add_media(1.0)
        return stack

    def trace(self):
        stack = self.build_stack()
        tr = stack.trace(self.source)
        irr = irradiance_from_trace(tr, detector_z=self.detector_z)
        self.trace_result = tr
        self.irradiance = irr
        return tr, irr

    def spot_size(self):
        """RMS ray spot radius vs detector plane, for auto-focus."""
        tr, _ = self.trace()
        pts0 = tr.hits[-1]
        zs = np.linspace(
            float(tr.hits[-1][:, 2].min()) - 5,
            float(tr.hits[-1][:, 2].max()) + 120, 121,
        )
        spots = []
        for z in zs:
            p, _ = tr.propagate_to(z)
            spots.append(np.sqrt(np.mean(p[:, 0] ** 2 + p[:, 1] ** 2)))
        spots = np.array(spots)
        return float(zs[np.argmin(spots)]), float(spots.min()), float(pts0.shape[0])
