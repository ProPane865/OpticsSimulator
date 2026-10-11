import numpy as np

from .vector import normalize

_EPS = 1e-12


class MeasurementPlane:
    """A planar detector parametrized as a graph over world XY.

    A point on the plane is ``(u, v, z(u, v))`` where ``u``/``v`` are the
    world X/Y coordinates and ``z`` follows the plane equation through
    ``origin`` with ``normal``. This requires a non-zero normal Z component.
    """

    def __init__(self, origin, normal, u_range, v_range, resolution):
        self.o = np.asarray(origin, dtype=float)
        if self.o.shape != (3,):
            raise ValueError("Origin must be array of shape (3,)")
        if not np.all(np.isfinite(self.o)):
            raise ValueError("Origin must be finite")

        self.n = normalize(normal)
        if abs(self.n[2]) <= _EPS:
            raise ValueError(
                "Measurement plane normal must have a non-zero Z component"
            )

        self.ru, self.rv = int(resolution[0]), int(resolution[1])
        if self.ru < 1 or self.rv < 1:
            raise ValueError("Resolution must be at least 1 in each direction")

        self.u_min, self.u_max = float(u_range[0]), float(u_range[1])
        self.v_min, self.v_max = float(v_range[0]), float(v_range[1])
        if not (self.u_max > self.u_min and self.v_max > self.v_min):
            raise ValueError("Parameter ranges must be increasing")

        self.grid = np.zeros((self.rv, self.ru), dtype=float)

    @property
    def interval_u(self):
        return self.u_max - self.u_min

    @property
    def interval_v(self):
        return self.v_max - self.v_min

    @property
    def bin_area(self):
        return (self.interval_u / self.ru) * (self.interval_v / self.rv)

    def point(self, u, v):
        """World-space point on the plane for parameters (u, v)."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        nxnz = self.n[0] / self.n[2]
        nynz = self.n[1] / self.n[2]
        z = self.o[2] - nxnz * (u - self.o[0]) - nynz * (v - self.o[1])
        return np.stack(np.broadcast_arrays(u, v, z), axis=-1)

    def intersect(self, ray):
        d = ray.direction
        ro = ray.origin
        if not (np.all(np.isfinite(ro)) and np.all(np.isfinite(d))):
            return None
        if abs(self.n[2]) <= _EPS or abs(d[2]) <= _EPS:
            return None

        nxnz = (self.n[0] / self.n[2])
        nynz = (self.n[1] / self.n[2])
        A = np.asarray([
            [-1, 0, d[0]],
            [0, -1, d[1]],
            [nxnz, nynz, d[2]],
        ])
        b = np.asarray([-ro[0], -ro[1], nxnz*self.o[0] + nynz*self.o[1] + self.o[2] - ro[2]])
        try:
            u, v, t = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        if not np.isfinite(t) or t <= 0.0:
            return None
        if not (np.isfinite(u) and np.isfinite(v)):
            return None

        return ray.at(t), float(u), float(v)

    def contains(self, u, v):
        """Boolean mask for parameters inside the plane extent."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return (
            (u >= self.u_min)
            & (u <= self.u_max)
            & (v >= self.v_min)
            & (v <= self.v_max)
        )

    def discretize(self, u, v):
        """Map parameters to integer bin indices (i, j)."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        i = np.floor((u - self.u_min) / self.interval_u * self.ru)
        j = np.floor((v - self.v_min) / self.interval_v * self.rv)

        i = np.clip(i, 0, self.ru - 1).astype(int)
        j = np.clip(j, 0, self.rv - 1).astype(int)

        return i, j

    def reset(self):
        self.grid[:] = 0.0

    def accumulate(self, u, v, power):
        """Add ray power into bins for parameters inside the plane extent."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        v = np.atleast_1d(np.asarray(v, dtype=float))
        power = np.atleast_1d(np.asarray(power, dtype=float))

        valid = self.contains(u, v) & np.isfinite(power)
        if not np.any(valid):
            return 0.0

        i, j = self.discretize(u[valid], v[valid])
        np.add.at(self.grid, (j, i), power[valid])
        return float(np.sum(power[valid]))

    def measure(self, rays, elements=None):
        """Trace rays through ``elements`` and accumulate power on the grid."""
        self.reset()

        trace = None
        if elements:
            from .refraction import trace_stack

            trace = lambda ray: trace_stack(ray, elements)

        total = 0.0
        for ray in rays:
            if trace is not None:
                ray = trace(ray)
                if ray is None:
                    continue
            hit = self.intersect(ray)
            if hit is None:
                continue
            _, u, v = hit
            total += self.accumulate(u, v, ray.power)

        return total

    def irradiance(self):
        """Power per unit true plane area for each bin."""
        area = self.bin_area / abs(self.n[2])
        return self.grid / area
