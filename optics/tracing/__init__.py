"""Ray tracing through refractive elements.

Layering (each module may only import from modules left of it):
``kernels`` <- ``context`` <- ``stack``, with ``element`` depending on no
tracing module. Internal imports always target the defining submodule, never
this package ``__init__``, so the trace call graph stays acyclic.
"""

from .context import build_trace_context, trace_segments_batch
from .element import RefractiveElement
from .stack import trace_stack_path

__all__ = [
    "RefractiveElement",
    "build_trace_context",
    "trace_segments_batch",
    "trace_stack_path",
]
