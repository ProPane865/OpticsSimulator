"""3D viewport built on VisPy scene visuals (meshes, lines, images) + orbit camera.

Surfaces, rays, the source pattern and the detector irradiance map are drawn
with VisPy ``Mesh`` / ``LinePlot`` / ``Image`` visuals inside a ``SceneCanvas``
whose native Qt widget is embedded by the application. Camera orbit / pan /
zoom is driven by the ``TurntableCamera`` through mouse handlers wired to the
canvas event stream.
"""

from __future__ import annotations

import numpy as np

from optics import rendering, ConicSurface, Plane


_VISPY_APP_READY = False


def _ensure_vispy_backend():
    global _VISPY_APP_READY
    if not _VISPY_APP_READY:
        import vispy.app
        vispy.app.use_app('PySide6')
        _VISPY_APP_READY = True


class Viewport:
    """VisPy-backed interactive viewport."""

    def __init__(self, parent=None):
        _ensure_vispy_backend()
        from vispy.scene import SceneCanvas
        import vispy.scene as S
        from vispy.scene import visuals as visuals
        from vispy.visuals.transforms.linear import MatrixTransform

        self.visuals = visuals
        self.MatrixTransform = MatrixTransform
        self.canvas = SceneCanvas(show=False, keys='interactive', parent=parent)
        self.canvas.bgcolor = (0.06, 0.07, 0.09, 1.0)
        self._view = self.canvas.central_widget.add_view()
        self.camera = S.TurntableCamera(
            fov=45.0, elevation=22.0, azimuth=30.0, distance=240.0,
        )
        self._view.camera = self.camera

        self.scene = None
        self.trace_result = None
        self.irradiance = None
        self._meshes = []
        self._axis_line = None
        self._ray_line = None
        self._src_img = None
        self._det_img = None
        self._mouse = None
        self._last = None

        self._connect_mouse()
        self._fit()
        self.refresh()

    # --- embedding --------------------------------------------------------
    @property
    def native(self):
        """The underlying Qt widget, to be set as the central widget."""
        return self.canvas.native

    # --- scene ------------------------------------------------------------
    def set_scene(self, scene):
        self.scene = scene
        self._fit()
        self.refresh()

    def _fit(self):
        if self.scene is None:
            return
        zs = ([self.scene.source_origin[2]]
              + [s["z0"] for s in self.scene.surfaces]
              + [self.scene.detector_z])
        zs = [z for z in zs if z is not None]
        lo, hi = min(zs), max(zs)
        self.camera.translate = np.array(
            [0.0, 0.0, 0.5 * (lo + hi)], dtype=np.float32)
        self.camera.distance = max(10.0, (hi - lo) * 2.5)

    def refresh(self):
        if self.scene is None:
            return
        self.trace_result, self.irradiance = self.scene.trace()
        self._build_geometry()
        self.canvas.update()

    # --- geometry ---------------------------------------------------------
    def _detach(self):
        for v in (list(self._meshes) +
                  [self._axis_line, self._ray_line, self._src_img, self._det_img]):
            if v is not None:
                v.parent = None
        self._meshes = []
        self._axis_line = self._ray_line = self._src_img = self._det_img = None

    def _build_geometry(self):
        self._detach()
        tr = self.trace_result
        scene = self.scene
        visuals = self.visuals

        for s in scene.surfaces:
            if s["kind"] == "conic":
                surf = ConicSurface(
                    s["name"], radius=s["radius"], conicity=s["conicity"],
                    z0=s["z0"], aperture=s["aperture"])
            else:
                surf = Plane(s["name"], z0=s["z0"], aperture=s["aperture"])
            ap = s.get("aperture") or 60.0
            verts, faces = rendering.surface_mesh(surf, min(ap, 60.0), n=48)
            color = np.broadcast_to(
                np.array([0.7, 0.8, 0.9, 0.55], np.float32),
                (len(verts), 4)).copy()
            mesh = visuals.Mesh(
                vertices=verts, faces=faces, vertex_colors=color,
                parent=self._view.scene)
            mesh.set_gl_state(preset='translucent', depth_mask=False)
            self._meshes.append(mesh)

        self._build_axes()
        self._build_rays(tr)
        self._build_textures()

    def _build_axes(self):
        L = 30.0
        o = np.array([0, 0, 0], np.float32)
        pts = np.zeros((8, 3), np.float32)
        pts[0] = o; pts[1] = [L, 0, 0]; pts[2] = np.nan
        pts[3] = o; pts[4] = [0, L, 0]; pts[5] = np.nan
        pts[6] = o; pts[7] = [0, 0, L]
        cols = np.array([
            [1, 0, 0, 1], [1, 0, 0, 0.4],
            [1, 0, 0, 1],
            [0, 1, 0, 1], [0, 1, 0, 0.4],
            [0, 1, 0, 1],
            [0, 0, 1, 1], [0, 0, 1, 0.4],
        ], np.float32)
        self._axis_line = self.visuals.LinePlot(
            data=pts, color=cols, connect='strip', parent=self._view.scene)

    @staticmethod
    def _ray_segment_colors(segs_per_ray, blocked, n_segs):
        cols = np.broadcast_to(np.array([[0.95, 0.7, 0.2, 0.85]], np.float32), (n_segs, 4)).copy()
        if blocked is not None:
            cols[np.repeat(blocked, segs_per_ray)] = [0.5, 0.5, 0.55, 0.85]
        return cols

    def _build_rays(self, tr):
        segs, _ = rendering.ray_polylines(tr, detector_z=self.scene.detector_z)
        n_rays = tr.n_rays
        segs_per_ray = tr.n_surfaces + 1
        segs = segs.reshape(n_rays, segs_per_ray, 2, 3)
        n_seg_total = n_rays * segs_per_ray
        seg_cols = self._ray_segment_colors(
            segs_per_ray, tr.blocked, n_seg_total)
        per_ray = segs_per_ray * 2
        v_all = segs.reshape(n_rays, per_ray, 3)
        c_all = np.repeat(seg_cols, 2, axis=0).reshape(n_rays, per_ray, 4)
        sep = np.full((n_rays, 1, 3), np.nan, np.float32)
        sep_c = seg_cols[::segs_per_ray].reshape(n_rays, 1, 4)
        pts = np.concatenate([v_all, sep], axis=1).reshape(-1, 3)
        cols = np.concatenate([c_all, sep_c], axis=1).reshape(-1, 4)
        self._ray_line = self.visuals.LinePlot(
            data=pts, color=cols, connect='strip', parent=self._view.scene)

    def _build_textures(self):
        src_corner0, src_span = self._source_span()
        src_shape = self.scene.source.shape
        self._src_img = self.visuals.Image(
            data=self._with_alpha(self._gray(self.scene.pattern), 0.9),
            parent=self._view.scene)
        self._src_img.transform = self.MatrixTransform(
            self._plane_transform(src_corner0, src_span, src_shape))
        self._src_img.set_gl_state(preset='translucent', depth_mask=False)

        det = self._detector_corners()
        if det is not None:
            corner0, span, shape = det
            self._det_img = self.visuals.Image(
                data=self._with_alpha(self._irradiance_texture(), 0.6),
                parent=self._view.scene)
            self._det_img.transform = self.MatrixTransform(
                self._plane_transform(corner0, span, shape))
            self._det_img.set_gl_state(preset='translucent', depth_mask=False)

    @staticmethod
    def _plane_transform(corner0, span, shape):
        h, w = shape[0], shape[1]
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = span[0] / w if w else 0.0
        m[1, 1] = span[1] / h if h else 0.0
        m[2, 2] = 1.0
        m[0, 3], m[1, 3], m[2, 3] = float(corner0[0]), float(corner0[1]), float(corner0[2])
        return m

    @staticmethod
    def _with_alpha(rgb, alpha):
        rgb = np.asarray(rgb, np.float32)
        h, w = rgb.shape[0], rgb.shape[1]
        rgba = np.empty((h, w, 4), np.float32)
        rgba[..., :3] = rgb
        rgba[..., 3] = alpha
        return rgba

    @staticmethod
    def _gray(pat):
        p = np.asarray(pat, np.float32)
        hi = float(p.max()) or 1.0
        g = p / hi
        return np.stack([g, g, g], axis=-1).astype(np.float32)

    def _irradiance_texture(self):
        img = self.irradiance["irradiance"]
        flat = img.ravel()
        valid = flat[np.isfinite(flat)]
        if valid.size == 0:
            return np.zeros((img.shape[0], img.shape[1], 3), np.float32)
        lo = np.percentile(valid, 1)
        hi = max(np.percentile(valid, 99.5), lo + 1e-9)
        logv = np.log1p(np.clip(img - lo, 0, None)) / np.log1p(hi - lo)
        logv = np.clip(logv, 0, 1)
        return np.stack([np.ones_like(logv), logv, np.zeros_like(logv)], axis=-1).astype(np.float32)

    def _source_span(self):
        src = self.scene.source
        pu, pv = src.pitch
        ox, oy, oz = src.origin
        hw, hh = src.shape[1] * pu / 2, src.shape[0] * pv / 2
        return np.array([ox - hw, oy - hh, oz], np.float32), (2.0 * hw, 2.0 * hh)

    def _detector_corners(self):
        if self.irradiance is None:
            return None
        xy = self.irradiance["output_xy"]
        xmin, xmax = float(xy[:, :, 0].min()), float(xy[:, :, 0].max())
        ymin, ymax = float(xy[:, :, 1].min()), float(xy[:, :, 1].max())
        pad = 0.08 * max(xmax - xmin, ymax - ymin, 1e-6)
        xmin -= pad; xmax += pad; ymin -= pad; ymax += pad
        z = self.scene.detector_z
        corner0 = np.array([xmin, ymin, z], np.float32)
        span = (xmax - xmin, ymax - ymin)
        shape = self.irradiance["irradiance"].shape
        return corner0, span, shape

    # --- mouse ------------------------------------------------------------
    def _connect_mouse(self):
        ev = self.canvas.events
        ev.mouse_press.connect(self._on_mouse_press)
        ev.mouse_move.connect(self._on_mouse_move)
        ev.mouse_release.connect(self._on_mouse_release)
        ev.mouse_wheel.connect(self._on_mouse_wheel)

    def _on_mouse_press(self, e):
        if e.button == 1:
            self._mouse = "orbit"
        elif e.button in (2, 3):
            self._mouse = "pan"
        p = e.pos
        self._last = (p[0], p[1])

    def _on_mouse_move(self, e):
        if self._mouse is None or e.pos is None:
            return
        p = e.pos
        x, y = p[0], p[1]
        dx = x - self._last[0]
        dy = y - self._last[1]
        self._last = (x, y)
        if self._mouse == "orbit":
            self.camera.azimuth += 0.4 * dx
            self.camera.elevation = np.clip(
                self.camera.elevation - 0.4 * dy, -88, 88)
            self.canvas.update()
        elif self._mouse == "pan":
            scale = self.camera.distance / 200.0
            t = np.asarray(self.camera.translate, dtype=np.float32).copy()
            t[0] -= 0.3 * dx * scale
            t[1] += 0.3 * dy * scale
            self.camera.translate = t
            self.canvas.update()

    def _on_mouse_release(self, e):
        self._mouse = None

    def _on_mouse_wheel(self, e):
        dy = e.delta[1] if e.delta else 0.0
        self.camera.distance = max(10.0, self.camera.distance * (1 - 0.05 * dy))
        self.canvas.update()
