import numpy as np
import sympy as sp
import json

class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin
        self.direction = direction / np.linalg.norm(direction)

class Surface:
    def implicit(self, p):
        """Return F(x,y,z)."""
        raise NotImplementedError

    def normal(self, p):
        """Return surface normal at p."""
        raise NotImplementedError

    def parameterize(self, u, v):
        """Return points for visualization."""
        raise NotImplementedError

    @property
    def parameter_range(self):
        raise NotImplementedError

class RefractiveElement:
    def __init__(self, schema: str, orientation=np.array([0, 0, 1])):
        self.sch = json.loads(schema)

        x, y = sp.symbols("x y")

        self.surface1 = self._compile(self.sch["surface1"]["equation"], x, y)
        self.surface2 = self._compile(self.sch["surface2"]["equation"], x, y)

        self.eq1 = self.surface1[0]
        self.eq2 = self.surface2[0]

        self.n = self.sch["material"]["refractive_index"]

        self.orientation = orientation / np.linalg.norm(orientation)

        original = np.array([0, 0, 1])
        new = self.orientation

        axis = np.cross(original, new)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-12:
            if np.inner(original, new) < 0:
                self.r_matrix = np.array([[1.0, 0.0, 0.0],
                                          [0.0, -1.0, 0.0],
                                          [0.0, 0.0, -1.0]])
            else:
                self.r_matrix = np.eye(3, 3)
        else:
            axis_mat = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])

            self.r_matrix = np.eye(3, 3) + axis_mat + np.matmul(axis_mat, axis_mat) * ((1 - np.inner(original, new)) / axis_norm**2)

    @staticmethod
    def _compile(equation: str, x: sp.Symbol, y: sp.Symbol) -> tuple:
        expr = sp.parse_expr(equation)
        return (
            sp.lambdify([x, y], expr),
            sp.lambdify([x, y], sp.diff(expr, x)),
            sp.lambdify([x, y], sp.diff(expr, y)),
        )

    def _intersect(self, surface, ray: Ray) -> np.ndarray | None:
        f, fx, fy = surface
        o = self.r_matrix.T @ np.asarray(ray.origin, dtype=float)
        d = self.r_matrix.T @ np.asarray(ray.direction, dtype=float)

        length = np.linalg.norm(o) + 2.0
        ts = np.linspace(-length, length, 401)
        h = np.empty(len(ts))
        with np.errstate(invalid="ignore", divide="ignore"):
            for i, t in enumerate(ts):
                p = o + t * d
                z = f(p[0], p[1])
                h[i] = z - p[2] if np.isfinite(z) else np.nan

        finite = np.where(np.isfinite(h))[0]
        if len(finite) == 0:
            return None

        candidates = ts[np.isfinite(h) & (np.abs(h) <= 0.05) & (ts > 0.0)]
        if len(candidates):
            t0 = candidates[0]
        else:
            t0 = ts[finite[np.argmin(np.abs(h[finite]))]]

        p0 = o + t0 * d
        xg, yg = p0[0], p0[1]

        for _ in range(50):
            with np.errstate(invalid="ignore", divide="ignore"):
                fg = f(xg, yg)
                fug = fx(xg, yg)
                fvg = fy(xg, yg)
            if not (np.isfinite(fg) and np.isfinite(fug) and np.isfinite(fvg)):
                return None

            A = np.array([[d[0], -1.0, 0.0],
                          [d[1], 0.0, -1.0],
                          [d[2], -fug, -fvg]])
            b = np.array([-o[0], -o[1], -o[2] + fg - fug * xg - fvg * yg])
            try:
                t, u, v = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return None
            if not (np.isfinite(t) and np.isfinite(u) and np.isfinite(v)):
                return None

            if np.hypot(u - xg, v - yg) < 1e-12 * max(1.0, abs(u), abs(v)):
                if t <= 0.0:
                    return None
                zf = f(u, v)
                if not np.isfinite(zf):
                    return None
                return self.r_matrix @ np.array([u, v, zf])
            xg, yg = u, v

        return None

    def get_normal(self, surface, pos: np.ndarray) -> np.ndarray:
        p = self.r_matrix.T @ np.asarray(pos, dtype=float)
        x, y = float(p[0]), float(p[1])
        fx, fy = surface[1], surface[2]
        n = -np.array([-fx(x, y), -fy(x, y), 1.0])
        norm = np.linalg.norm(n)
        if not np.isfinite(norm):
            return None
        unit_n = n / norm
        unit_n_rot = np.matmul(self.r_matrix, unit_n)
        return unit_n_rot

    def refract(self, ray: Ray) -> Ray:
        p1 = self._intersect(self.surface1, ray)
        n1 = self.get_normal(self.surface1, p1)
        eta1 = 1 / self.n
        transfer1 = eta1 * np.eye(3, 3) + ((np.sqrt(1 - eta1**2 * (1 - np.inner(n1, ray.direction)**2)) / np.inner(n1, ray.direction)) - eta1) * np.outer(n1, n1)
        d1 = transfer1 @ ray.direction
        r1 = Ray(p1, d1)

        p2 = self._intersect(self.surface2, r1)
        n2 = self.get_normal(self.surface2, p2)
        eta2 = self.n
        transfer2 = eta2 * np.eye(3, 3) + ((np.sqrt(1 - eta2**2 * (1 - np.inner(n2, d1)**2)) / np.inner(n2, d1)) - eta2) * np.outer(n2, n2)
        d2 = transfer2 @ d1

        return Ray(p2, d2)

def check_on_axis(label, element, orientation):
    s = orientation / np.linalg.norm(orientation)
    ray = Ray(2.0 * s, -s)

    p1 = element._intersect(element.surface1, ray)
    assert p1 is not None, f"{label}: no entry intersection"
    assert np.isclose(np.linalg.norm(p1), 1.0, atol=1e-6), f"{label}: entry point not on sphere: {p1}"
    assert (element.r_matrix.T @ p1)[2] > 0.0, f"{label}: entry point not on surface1 hemisphere: {p1}"

    out = element.refract(ray)
    assert out.origin is not None, f"{label}: no exit point"
    assert np.allclose(out.direction, -s, atol=1e-6), f"{label}: on-axis ray deflected: {out.direction}"
    assert np.isclose(np.linalg.norm(out.origin), 1.0, atol=1e-6), f"{label}: exit point not on sphere: {out.origin}"

    print(f"{label}: origin={out.origin}, direction={out.direction}")


if __name__ == "__main__":
    with open("tests/test_geometry.json", "r") as f:
        schema = f.read()

    check_on_axis("default orientation", RefractiveElement(schema), np.array([0.0, 0.0, 1.0]))
    check_on_axis("tilted orientation", RefractiveElement(schema, np.array([1.0, 0.5, 1.0])), np.array([1.0, 0.5, 1.0]))