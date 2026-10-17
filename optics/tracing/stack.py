"""Rendering path building on top of the native stack-trace kernels."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..primitive import Ray
from .context import _trace_stack_path_fast, build_trace_context

if TYPE_CHECKING:
    from .element import RefractiveElement


def trace_stack_path(
    ray: Ray, elements: Sequence["RefractiveElement"], ray_length: float = 3.0
) -> "tuple[list[tuple[np.ndarray, np.ndarray, str]], Ray | None]":
    """Return ``(segments, final_ray)`` for rendering.

    Each segment is a ``(start, end, kind)`` tuple where ``kind`` is
    ``"incident"`` before the first element, ``"internal"`` inside an
    element, and ``"exit"`` after leaving one.

    Runs through the native stack kernel; an empty stack yields a single
    straight incident segment. A surface the kernels cannot encode raises
    when its descriptor is built.
    """
    ctx = build_trace_context(elements)
    if ctx is not None:
        return _trace_stack_path_fast(ray, ctx, ray_length, len(elements))
    end = ray.origin + ray_length * ray.direction
    return [(np.asarray(ray.origin, dtype=float), end, "incident")], ray
