"""Native trace-kernel entry points: element first-hit and stack walking.

The stack-walking kernels themselves run in C++ (``optics._native``); this
module keeps the segment-capacity constants and the Python-facing wrappers
with contiguous-array handling.
"""

from typing import Any

import numpy as np

from .. import _native
from .. import constants as _c

_MAX_INTERNAL_BOUNCES = _c.MAX_INTERNAL_BOUNCES
_MAX_SEGS_PER_ELEMENT = 2 * _MAX_INTERNAL_BOUNCES + 3


def trace_stack_segments_njit(
    stack: Any,
    o: np.ndarray,
    d: np.ndarray,
    ray_length: float,
    starts: np.ndarray,
    ends: np.ndarray,
    kinds: np.ndarray,
) -> "tuple[int, bool, np.ndarray, np.ndarray]":
    """Walk one ray through the stack; returns ``(count, exited, fo, fd)``.

    ``starts``/``ends`` are ``(cap, 3)`` buffers, ``kinds`` is ``(cap,)``
    int8; the per-segment kind codes are 0 incident / 1 internal / 2 exit.
    """
    return _native.trace_stack_segments(
        stack,
        np.ascontiguousarray(o, dtype=float),
        np.ascontiguousarray(d, dtype=float),
        float(ray_length),
        starts,
        ends,
        kinds,
    )


def trace_stack_to_plane_njit(
    stack: Any,
    o: np.ndarray,
    d: np.ndarray,
    po: np.ndarray,
    pn: np.ndarray,
    eps: float,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    out_o: np.ndarray,
    out_d: np.ndarray,
) -> bool:
    """Trace through the stack until the plane region is reached.

    Writes the ray that reaches the measurement-plane extent into
    ``out_o``/``out_d``; ``True`` when such a ray was found.
    """
    return bool(
        _native.trace_stack_to_plane(
            stack,
            np.ascontiguousarray(o, dtype=float),
            np.ascontiguousarray(d, dtype=float),
            np.ascontiguousarray(po, dtype=float),
            np.ascontiguousarray(pn, dtype=float),
            float(eps),
            float(u_min),
            float(u_max),
            float(v_min),
            float(v_max),
            out_o,
            out_d,
        )
    )
