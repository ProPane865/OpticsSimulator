"""Ray tracing through refractive elements.

Layering (each module may only import from modules left of it):
``direction`` <- ``kernels`` <- ``context`` <- ``stack``, with ``element``
depending only on ``direction``.
Internal imports always target the defining submodule, never this package
``__init__``, so the trace call graph stays acyclic.
"""

from .context import build_trace_context, trace_segments_batch
from .direction import refract_or_reflect, trace_direction
from .element import RefractiveElement, TraceResult
from .stack import (
    internal_rays,
    trace_stack,
    trace_stack_path,
    traverse_stack,
)

__all__ = [
    "RefractiveElement",
    "TraceResult",
    "build_trace_context",
    "internal_rays",
    "refract_or_reflect",
    "trace_direction",
    "trace_segments_batch",
    "trace_stack",
    "trace_stack_path",
    "traverse_stack",
]
