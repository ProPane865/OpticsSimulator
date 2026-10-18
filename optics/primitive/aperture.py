from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

from .. import constants as _c

if TYPE_CHECKING:
    from .surface import Surface

# The aperture rim depends only on the surface's *local* geometry (expressions,
# parameter ranges) plus the aperture circle and sample count -- not its world
# transform. Caching the local rim by a geometry signature lets identical or
# freshly rebuilt surfaces reuse the (expensive) parameter-inversion solves and
# re-apply only the world transform.
_RIM_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_RIM_CACHE_MAX = 64


def _rim_key(surface: "Surface", aperture: "CircularAperture", n: int) -> tuple:
    return (
        surface.x_expr,
        surface.y_expr,
        surface.z_expr,
        tuple(surface.u_range),
        tuple(surface.v_range),
        float(aperture.radius),
        float(aperture.center[0]),
        float(aperture.center[1]),
        int(n),
    )


class Aperture:
    def contains(self, surface: "Surface", u: float, v: float) -> bool:
        raise NotImplementedError

    def mask(
        self,
        surface: "Surface",
        U: np.ndarray,
        V: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        """Vectorized aperture membership over parameter arrays ``U``/``V``.

        The default implementation evaluates :meth:`contains` point by point;
        subclasses override it with a vectorized form for speed. ``X``/``Y``/``Z``
        are the local-space surface points, available for apertures that test
        geometry directly.
        """
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        U, V = np.broadcast_arrays(U, V)
        out = np.array(
            [bool(self.contains(surface, u, v)) for u, v in zip(U.ravel(), V.ravel())],
            dtype=bool,
        )
        return out.reshape(U.shape)

    def boundary(self, surface: "Surface", n: int = _c.APERTURE_BOUNDARY_N) -> np.ndarray:
        raise NotImplementedError

    def native_params(self) -> "tuple[float, float, float] | None":
        """Circular slots ``(radius, cx, cy)`` for the native kernels.

        ``None`` for aperture types the kernels cannot encode; building the
        surface's native object then fails, and the stack cannot trace.
        """
        return None


class CircularAperture(Aperture):
    radius: float
    center: np.ndarray

    def __init__(self, radius: float, center: "tuple[float, float] | list[float]" = (0.0, 0.0)) -> None:
        self.radius = float(radius)
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("Aperture radius must be finite and positive")
        self.center = np.array(center, dtype=float)
        if self.center.shape != (2,):
            raise ValueError("Aperture center must be an array of shape (2,)")
        if not np.all(np.isfinite(self.center)):
            raise ValueError("Aperture center must be finite")

    def contains(self, surface: "Surface", u: float, v: float) -> bool:
        p = surface.evaluate_local(u, v)

        if not np.all(np.isfinite(p)):
            return False

        dx = p[0] - self.center[0]
        dy = p[1] - self.center[1]

        return dx ** 2 + dy ** 2 <= self.radius ** 2 + _c.APERTURE_EPS

    def mask(
        self,
        surface: "Surface",
        U: np.ndarray,
        V: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        finite = (
            np.isfinite(X)
            & np.isfinite(Y)
            & np.isfinite(Z)
        )

        dx = X - self.center[0]
        dy = Y - self.center[1]

        return (
            finite
            & (dx ** 2 + dy ** 2 <= self.radius ** 2 + _c.APERTURE_EPS)
        )

    def boundary(self, surface: "Surface", n: int = _c.APERTURE_BOUNDARY_N) -> np.ndarray:
        key = _rim_key(surface, self, n)
        cached = _RIM_CACHE.get(key)
        if cached is not None:
            _RIM_CACHE.move_to_end(key)
            local = cached
        else:
            # Invert the rim circle onto the surface in a single native pass
            # on the persistent surface (continuation seeding, matching the
            # previous per-point loop).
            local = surface.native_surface.rim(
                self.radius,
                float(self.center[0]),
                float(self.center[1]),
                int(n),
            )
            _RIM_CACHE[key] = local
            _RIM_CACHE.move_to_end(key)
            while len(_RIM_CACHE) > _RIM_CACHE_MAX:
                _RIM_CACHE.popitem(last=False)

        return surface.transform.points_to_world(local)

    def native_params(self) -> "tuple[float, float, float]":
        return self.radius, float(self.center[0]), float(self.center[1])