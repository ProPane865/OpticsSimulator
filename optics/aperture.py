import numpy as np

class Aperture:
    @staticmethod
    def from_spec(spec):
        if spec is None:
            return None
        if isinstance(spec, Aperture):
            return spec
        if isinstance(spec, dict):
            kind = str(spec.get("type", "circular")).lower()
            if kind == "circular":
                return CircularAperture(
                    spec["radius"],
                    spec.get("center", (0.0, 0.0)),
                )
            raise ValueError(f"Unknown aperture type: {kind}")
        raise TypeError(f"Unsupported aperture spec: {spec!r}")

    def contains(self, surface, u, v) -> bool:
        raise NotImplementedError

    def mask(self, surface, U, V, X, Y, Z):
        raise NotImplementedError

    def boundary(self, surface, n=128) -> np.ndarray:
        raise NotImplementedError

class CircularAperture(Aperture):
    def __init__(self, radius, center=(0.0, 0.0)):
        self.radius = float(radius)
        self.center = np.array(center, dtype=float)

    def contains(self, surface, u, v) -> bool:
        p = surface.evaluate_local(u, v)

        if not np.all(np.isfinite(p)):
            return False

        dx = p[0] - self.center[0]
        dy = p[1] - self.center[1]

        return dx ** 2 + dy ** 2 <= self.radius ** 2 + 1e-12

    def mask(self, surface, U, V, X, Y, Z):
        finite = (
            np.isfinite(X)
            & np.isfinite(Y)
            & np.isfinite(Z)
        )

        dx = X - self.center[0]
        dy = Y - self.center[1]

        return (
            finite
            & (dx ** 2 + dy ** 2 <= self.radius ** 2)
        )

    def boundary(self, surface, n=128):
        theta = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
        targets = np.column_stack([
            self.center[0] + self.radius * np.cos(theta),
            self.center[1] + self.radius * np.sin(theta)
        ])

        points = []

        for x, y in targets:
            uv = surface.parameters_at_xy(x, y)

            if uv is None:
                raise ValueError(f"Aperture boundary point ({x}, {y}) does not lie on the surface")

            u, v = uv
            p = surface.point(u, v)

            if not np.all(np.isfinite(p)):
                raise ValueError("Non-finite point on aperture boundary")

            points.append(p)
        
        return np.array(points, dtype=float)