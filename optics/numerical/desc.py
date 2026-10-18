"""Native surface-descriptor access: evaluation, derivatives, membership.

The kernels themselves run in C++ (``optics._native``); this module builds the
descriptor from a duck-typed surface (compiled model, aperture, parameter
range, transform) and exposes the Python-level accessors the tracing/numerical
drivers call. A descriptor is a native object shared by every surface with the
same compiled model type, so there is no per-surface compilation cost left.
"""

from typing import Any

from .. import constants as _c
from ._kernels import build_descriptor


def build_surface_descriptor(surface: Any) -> Any:
    """Flatten a ``Surface`` into the native kernel descriptor.

    Raises :class:`ValueError` when the surface carries an aperture the
    kernels cannot encode (see
    :meth:`~optics.primitive.aperture.Aperture.native_params`).
    """
    if surface.aperture is None:
        ap_r, ap_cx, ap_cy = -1.0, 0.0, 0.0
    else:
        params = surface.aperture.native_params()
        if params is None:
            raise ValueError(
                "Aperture type is not supported by the native surface kernels"
            )
        ap_r, ap_cx, ap_cy = params
    (u_min, u_max), (v_min, v_max) = surface.parameter_range
    tu = _c.DOMAIN_TOL * max(1.0, u_max - u_min)
    tv = _c.DOMAIN_TOL * max(1.0, v_max - v_min)
    return build_descriptor(
        surface._model,
        u_min,
        u_max,
        v_min,
        v_max,
        tu,
        tv,
        ap_r,
        ap_cx,
        ap_cy,
        surface.transform.rotation,
        surface.transform.translation,
    )


def desc_eval(desc: Any, u: float, v: float) -> Any:
    """Local-space surface point at ``(u, v)``."""
    return desc.eval(float(u), float(v))


def desc_accepts(desc: Any, u: float, v: float) -> bool:
    """Domain + circular-aperture membership (``accepts_parameters``)."""
    return bool(desc.accepts(float(u), float(v)))


def desc_derivatives(desc: Any, u: float, v: float) -> "tuple[Any, Any]":
    """Analytic tangents with the interior pull-back / FD fallback."""
    ru, rv = desc.derivatives(float(u), float(v))
    return ru, rv
