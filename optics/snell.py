"""Vector form of Snell's law of refraction.

Given an incident unit ray ``d`` and a surface unit normal ``n``, the refracted
unit ray is::

    eta  = n1 / n2
    cosi = -n . d            (normal oriented against the incoming ray)
    sin2 = eta^2 * (1 - cosi^2)
    if sin2 > 1   -> total internal reflection
    cost = sqrt(1 - sin2)
    d'     = eta * d + (eta*cosi - cost) * n

All operations are vectorized over a batch of rays of shape (N, 3).
"""

from __future__ import annotations

import numpy as np


def refract(incident, normal, n1, n2):
    """Refract a batch of rays.

    Parameters
    ----------
    incident : (N, 3) array of (not necessarily unit) incident ray directions.
    normal : (N, 3) array of surface normals (not necessarily unit).
    n1, n2 : refractive indices of the incident and transmitted media.

    Returns
    -------
    refracted : (N, 3) unit refracted ray directions.
    cos_i : (N,) cosine of incidence angle.
    total_internal_reflection : (N,) bool flag.
    """
    incident = np.asarray(incident, dtype=float)
    normal = np.asarray(normal, dtype=float)

    din = np.sqrt(np.sum(incident * incident, axis=-1, keepdims=True))
    d = incident / np.maximum(din, 1e-30)

    nn = np.sqrt(np.sum(normal * normal, axis=-1, keepdims=True))
    n = normal / np.maximum(nn, 1e-30)

    cos_i = -np.sum(n * d, axis=-1)
    flip = cos_i < 0.0
    cos_i = np.where(flip, -cos_i, cos_i)
    n = np.where(flip[:, None], -n, n)

    eta = np.float64(n1) / np.float64(n2)
    sin2 = eta * eta * (1.0 - cos_i * cos_i)
    tir = sin2 > 1.0
    sin2 = np.minimum(sin2, 1.0)
    cos_t = np.sqrt(np.maximum(1.0 - sin2, 0.0))

    refracted = eta * d + (eta * cos_i - cos_t)[:, None] * n
    rnorm = np.sqrt(np.sum(refracted * refracted, axis=-1, keepdims=True))
    refracted = refracted / np.maximum(rnorm, 1e-30)
    return refracted, cos_i, tir
