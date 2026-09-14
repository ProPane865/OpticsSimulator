"""Optical surface definitions.

Surfaces are described by a sag function ``z = g(x, y)``. The surface normal is
obtained from the surface differential (gradient): for ``z = g(x, y)`` the
(unoriented) normal is ``(-dg/dx, -dg/dy, 1)``. Ray/surface intersection uses a
closed form for planes and a Newton-Raphson iteration for conic / polynomial
sag surfaces.
"""

from __future__ import annotations

import numpy as np

PLANE = 0
CONIC = 1
POLY = 2


def _arr(v):
    return np.asarray(v, dtype=float)


class Surface:
    """Base class for a refracting surface defined by a sag ``z = g(x, y)``."""

    kind = None

    def __init__(self, name: str = "surface", z0: float = 0.0):
        self.name = name
        self.z0 = float(z0)

    # --- geometry ---------------------------------------------------------
    def sag(self, x, y):
        raise NotImplementedError

    def gradient(self, x, y):
        """Return ``(gx, gy)`` partial derivatives of the sag."""
        raise NotImplementedError

    def normal(self, x, y):
        """Return the geometric unit normal ``(-gx, -gy, 1)`` normalized, shape (..., 3)."""
        gx, gy = self.gradient(x, y)
        gx, gy = _arr(gx), _arr(gy)
        z = np.ones_like(gx)
        n = np.stack([-gx, -gy, z], axis=-1)
        norm = np.sqrt(np.sum(n * n, axis=-1, keepdims=True))
        return n / norm

    def intersect(self, origins, dirs):
        """Intersect batched rays ``origins, dirs`` (each (N, 3)) with this surface.

        Returns ``(t, points)`` where ``t`` is (N,) path length and ``points`` is (N, 3).
        """
        raise NotImplementedError

    # --- numba integration ------------------------------------------------
    def numba_descriptor(self):
        """Return ``(kind, coeffs_1d)`` consumable by the JIT tracer."""
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r})"


class Plane(Surface):
    """Flat surface at ``z = z0``."""

    kind = PLANE

    def __init__(self, name: str = "plane", z0: float = 0.0, aperture=None):
        super().__init__(name, z0)
        self.aperture = float(aperture) if aperture is not None else None

    def sag(self, x, y):
        return np.full(_arr(x).shape, self.z0)

    def gradient(self, x, y):
        x = _arr(x)
        return np.zeros_like(x), np.zeros_like(x)

    def intersect(self, origins, dirs):
        origins = _arr(origins)
        dirs = _arr(dirs)
        dz = dirs[..., 2]
        nz = dz != 0
        t = np.full(origins.shape[0], np.nan)
        t[nz] = (self.z0 - origins[nz, 2]) / dz[nz]
        points = origins + t[:, None] * dirs
        return t, points

    def numba_descriptor(self):
        ap2 = self.aperture ** 2 if self.aperture is not None else -1.0
        return PLANE, np.array([self.z0, ap2], dtype=float)


class ConicSurface(Surface):
    """Conic surface of revolution (sag of revolution about the z-axis).

    ``R`` is the radius of curvature at the vertex (signed: positive when the
    center of curvature lies in +z from the vertex). ``k`` is the conicity
    (edge factor): k = 0 sphere, k = -1 parabola, k < -1 hyperbola, k > 0 ellipsoid.
    """

    kind = CONIC

    def __init__(self, name: str = "conic", radius: float = np.inf,
                 conicity: float = 0.0, z0: float = 0.0, aperture=None):
        super().__init__(name, z0)
        self.radius = float(radius)
        self.conicity = float(conicity)
        self.aperture = float(aperture) if aperture is not None else None
        self.curvature = 1.0 / self.radius if np.isfinite(self.radius) else 0.0

    def sag(self, x, y):
        x, y = _arr(x), _arr(y)
        r2 = x * x + y * y
        c = self.curvature
        k = self.conicity
        s = 1.0 - (1.0 - k) * c * c * r2
        sigma = np.sqrt(np.maximum(s, 0.0))
        D = 1.0 + sigma
        return c * r2 / D + self.z0

    def gradient(self, x, y):
        x, y = _arr(x), _arr(y)
        r2 = x * x + y * y
        c = self.curvature
        k = self.conicity
        s = 1.0 - (1.0 - k) * c * c * r2
        sigma = np.sqrt(np.maximum(s, 1e-15))
        D = 1.0 + sigma
        factor = (c / (D * D)) * (2.0 * D + (1.0 - k) * c * c * r2 / sigma)
        return factor * x, factor * y

    def intersect(self, origins, dirs, max_iter: int = 40, tol: float = 1e-12):
        origins = _arr(origins)
        dirs = _arr(dirs)
        return _newton_sag(
            origins, dirs,
            coeffs=np.array([self.curvature, self.conicity, self.z0], dtype=float),
            kind=CONIC, max_iter=max_iter, tol=tol,
        )

    def numba_descriptor(self):
        ap2 = self.aperture ** 2 if self.aperture is not None else -1.0
        return CONIC, np.array([self.curvature, self.conicity, self.z0, ap2], dtype=float)


class PolySagSurface(Surface):
    """General sag surface ``z = z0 + sum C[a,b] x^a y^b`` over a square grid."""

    kind = POLY

    def __init__(self, name: str = "poly", coeffs=None, order: int = 0, z0: float = 0.0):
        super().__init__(name, z0)
        self.order = int(order)
        if coeffs is None:
            coeffs = np.zeros((order + 1, order + 1))
        self.coeffs = _arr(coeffs)
        if self.coeffs.shape != (order + 1, order + 1):
            raise ValueError("coeffs must have shape (order+1, order+1)")

    def _eval(self, x, y, deriv):
        x, y = _arr(x), _arr(y)
        out = np.zeros_like(x)
        C = self.coeffs
        oa, ob = C.shape
        xa = np.ones_like(x)
        xa_prev = np.ones_like(x)
        for a in range(oa):
            ya = np.ones_like(y)
            ya_prev = np.ones_like(y)
            for b in range(ob):
                term = C[a, b] * xa * ya
                if deriv == 0:
                    out = out + term
                elif deriv == 1 and a > 0:
                    out = out + a * C[a, b] * xa_prev * ya
                elif deriv == 2 and b > 0:
                    out = out + b * C[a, b] * xa * ya_prev
                if b < ob - 1:
                    ya_prev = ya
                    ya = ya * y
            xa_prev = xa
            xa = xa * x
        return out

    def sag(self, x, y):
        return self._eval(x, y, 0) + self.z0

    def gradient(self, x, y):
        gx = self._eval(x, y, 1)
        gy = self._eval(x, y, 2)
        return gx, gy

    def intersect(self, origins, dirs, max_iter: int = 40, tol: float = 1e-12):
        origins = _arr(origins)
        dirs = _arr(dirs)
        flat = np.concatenate([[self.order, self.z0], self.coeffs.ravel()])
        return _newton_sag(origins, dirs, coeffs=flat, kind=POLY,
                           max_iter=max_iter, tol=tol)

    def numba_descriptor(self):
        flat = np.concatenate([[self.order, self.z0], self.coeffs.ravel()])
        return POLY, flat


def _newton_sag(origins, dirs, coeffs, kind, max_iter=40, tol=1e-12):
    """Intersect rays with a sag surface via Newton-Raphson on ``g(x(t),y(t)) - z(t)``."""
    origins = _arr(origins)
    dirs = _arr(dirs)
    n = origins.shape[0]
    x0, y0, z0r = origins[:, 0], origins[:, 1], origins[:, 2]
    dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    if kind == PLANE:
        zc = coeffs[0]
        nz = dz != 0
        t = np.full(n, np.nan)
        t[nz] = (zc - z0r[nz]) / dz[nz]
        points = origins + t[:, None] * dirs
        return t, points

    if kind == CONIC:
        c, k, zc = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    order = int(coeffs[0]) if kind == POLY else -1

    def sag_and_grad(px, py, pz):
        r2 = px * px + py * py
        if kind == CONIC:
            s = 1.0 - (1.0 - k) * c * c * r2
            sigma = np.sqrt(np.maximum(s, 1e-15))
            D = 1.0 + sigma
            z = c * r2 / D + zc
            f = (c / (D * D)) * (2.0 * D + (1.0 - k) * c * c * r2 / sigma)
            gx = f * px
            gy = f * py
        else:  # POLY
            z = zc
            gx = 0.0
            gy = 0.0
            C = coeffs[2:2 + (order + 1) * (order + 1)].reshape(order + 1, order + 1)
            xa = np.ones_like(px)
            xa_prev = np.ones_like(px)
            for a in range(order + 1):
                if a > 0:
                    xa_prev = xa
                    xa = xa * px
                ya = np.ones_like(py)
                ya_prev = np.ones_like(py)
                for b in range(order + 1):
                    term = C[a, b] * xa * ya
                    z = z + term
                    if a > 0:
                        gx = gx + a * C[a, b] * xa_prev * ya
                    if b > 0:
                        gy = gy + b * C[a, b] * xa * ya_prev
                    if b < order:
                        ya_prev = ya
                        ya = ya * py
        return z, gx, gy

    t = np.where(np.abs(dz) > 1e-12, (0.0 - z0r) / dz + 1e-3, np.ones(n))
    t = np.maximum(t, 1e-9)

    for _ in range(max_iter):
        z_surf, gx, gy = sag_and_grad(x0 + t * dx, y0 + t * dy, z0r + t * dz)
        f = z_surf - (z0r + t * dz)
        dfdt = gx * dx + gy * dy - dz
        with np.errstate(divide="ignore", invalid="ignore"):
            step = f / dfdt
        step = np.where(np.isfinite(step), np.clip(step, -1.0, 1.0), 0.0)
        t_new = t - step
        t_new = np.maximum(t_new, 1e-12)
        if np.max(np.abs(t_new - t)) < tol:
            t = t_new
            break
        t = t_new

    z_surf, gx, gy = sag_and_grad(x0 + t * dx, y0 + t * dy, z0r + t * dz)
    points = np.stack([x0 + t * dx, y0 + t * dy, z_surf], axis=-1)
    return t, points
