from typing import TYPE_CHECKING

import numpy as np

from .. import constants as _c

if TYPE_CHECKING:
    from .surface import Surface


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

    def njit_params(self) -> "tuple[float, float, float] | None":
        """Circular slots ``(radius, cx, cy)`` for the nopython kernels.

        ``None`` for aperture types the kernels cannot encode, which keeps
        the surface on the Python solver path.
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
        theta = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
        targets = np.column_stack([
            self.center[0] + self.radius * np.cos(theta),
            self.center[1] + self.radius * np.sin(theta)
        ])

        points = []
        last = None

        for x, y in targets:
            uv = surface.parameters_at_xy(x, y, seed=last)

            if uv is None:
                raise ValueError(f"Aperture boundary point ({x}, {y}) does not lie on the surface")

            u, v = uv
            last = uv
            p = surface.point(u, v)

            if not np.all(np.isfinite(p)):
                raise ValueError("Non-finite point on aperture boundary")

            points.append(p)
        
        return np.array(points, dtype=float)

    def njit_params(self) -> "tuple[float, float, float]":
        return self.radius, float(self.center[0]), float(self.center[1])