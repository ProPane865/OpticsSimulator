"""Output irradiance from a trace via flux conservation and the ray-map Jacobian.

Conservation of radiometric flux through a ray bundle gives::

    I_out * dA_out = I_in * dA_in  ->  I_out = I_in / |det J|

where ``J = d(output_xy) / d(input_uv)`` is the Jacobian of the map from input
plane coordinates to detector-plane coordinates. ``J`` is obtained by finite
differences of the sampled ray map. Near caustics ``|det J| -> 0``; those cells
are clamped / masked to avoid singularities.
"""

from __future__ import annotations

import numpy as np


def compute_jacobian(trace, detector_z):
    """Return the input->detector Jacobian determinant grid (H, W)."""
    points, _t = trace.propagate_to(detector_z)
    H, W = trace.grid_shape
    du, dv = trace.pitch

    U = trace.uv[:, 0].reshape(H, W)
    V = trace.uv[:, 1].reshape(H, W)
    X = points[:, 0].reshape(H, W)
    Y = points[:, 1].reshape(H, W)

    dXdU = np.gradient(X, du, axis=1)
    dXdV = np.gradient(X, dv, axis=0)
    dYdU = np.gradient(Y, du, axis=1)
    dYdV = np.gradient(Y, dv, axis=0)
    return dXdU * dYdV - dXdV * dYdU, X, Y, U, V


def irradiance_from_trace(trace, detector_z, eps=1e-10):
    """Compute output irradiance on the detector plane.

    Returns a dict with keys:
        ``irradiance`` (H, W), ``jacobian_det`` (H, W), ``valid`` (H, W) bool,
        ``output_xy`` (H, W, 2), ``input_uv`` (H, W, 2).
    """
    detJ, X, Y, U, V = compute_jacobian(trace, detector_z)
    H, W = trace.grid_shape
    I_in = np.asarray(trace.irradiance_input, dtype=float).reshape(H, W)
    points, t = trace.propagate_to(detector_z)
    Xf = X.reshape(-1)
    Yf = Y.reshape(-1)
    tf = t.reshape(-1)

    blocked = None if trace.blocked is None else trace.blocked.reshape(-1)
    valid = (
        np.isfinite(Xf) & np.isfinite(Yf) & np.isfinite(detJ.reshape(-1))
        & (tf > 0) & np.isfinite(I_in.reshape(-1))
    )
    if blocked is not None:
        valid = valid & (~blocked)
    valid = valid.reshape(H, W)

    detJ_abs = np.clip(np.abs(detJ), eps, None)
    I_out = I_in / detJ_abs
    I_out = np.where(valid, I_out, 0.0)

    return {
        "irradiance": I_out,
        "jacobian_det": detJ,
        "valid": valid,
        "output_xy": np.stack([X, Y], axis=-1),
        "input_uv": np.stack([U, V], axis=-1),
    }


def flux_conservation(trace, detector_z, tol=0.1):
    """Relative difference between emitted and captured power (diagnostic)."""
    I_in = np.asarray(trace.irradiance_input, dtype=float)
    emitted = float(np.sum(I_in) * trace.pitch[0] * trace.pitch[1])
    captured = float(np.sum(irradiance_from_trace(trace, detector_z)["irradiance"]))
    if emitted == 0:
        return 0.0
    return abs(captured - emitted) / emitted
