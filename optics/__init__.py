"""3D geometric optics simulator."""

from . import accel
from .irradiance import compute_jacobian, irradiance_from_trace, flux_conservation
from .rays import RayStack
from .source import IrradianceSource
from .surfaces import ConicSurface, Plane, PolySagSurface, Surface
from .stack import OpticalStack, TraceResult
from .snell import refract

__all__ = [
    "accel",
    "IrradianceSource",
    "OpticalStack",
    "TraceResult",
    "RayStack",
    "Surface",
    "Plane",
    "ConicSurface",
    "PolySagSurface",
    "refract",
    "irradiance_from_trace",
    "compute_jacobian",
    "flux_conservation",
]
