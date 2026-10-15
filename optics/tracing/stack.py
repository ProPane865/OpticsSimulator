"""Geometric-order traversal of an element stack and path building."""

from collections.abc import Callable, Iterator, Sequence

import numpy as np

from ..primitive import Ray
from .context import _trace_stack_path_fast, build_trace_context
from .element import RefractiveElement, TraceResult


def _nearest_entry(
    elements: Sequence["RefractiveElement"], ray: Ray, t_max: float = np.inf
) -> "tuple[float, RefractiveElement] | None":
    best: tuple[float, RefractiveElement] | None = None
    for element in elements:
        entry = element.first_hit(ray)
        if entry is None or entry[0] >= t_max:
            continue
        if best is None or entry[0] < best[0]:
            best = (entry[0], element)
    return best


def traverse_stack(
    ray: Ray,
    elements: Sequence[RefractiveElement],
    limit_fn: "Callable[[Ray], float] | None" = None,
) -> Iterator[tuple[RefractiveElement, TraceResult]]:
    """Yield ``(element, TraceResult)`` visiting elements in geometric order.

    Iteration stops when no remaining element is entered before
    ``limit_fn(ray)`` (if given), when no element is hit, or when a ray
    is absorbed inside an element. An element already containing the ray
    is visited at its nearest face hit and traced from its interior.
    """
    current = ray
    remaining = list(elements)
    while remaining:
        t_max = np.inf if limit_fn is None else float(limit_fn(current))
        best = _nearest_entry(remaining, current, t_max)
        if best is None:
            return
        remaining.remove(best[1])
        result = best[1].trace(current)
        yield best[1], result
        if result.outgoing_ray is None:
            return
        current = result.outgoing_ray


def internal_rays(result: TraceResult) -> "Iterator[tuple[Ray, np.ndarray | None]]":
    """Yield ``(ray, endpoint)`` for each segment traveled inside an element.

    ``endpoint`` is ``None`` when the internal ray leaves the element's
    interior without a recorded hit (open side or bounce limit reached).
    """
    path = result.internal_path
    for i, segment in enumerate(path):
        if i + 1 < len(path):
            yield segment, path[i + 1].origin
        elif result.exit_hit is not None:
            yield segment, result.exit_hit.point
        else:
            yield segment, None


def trace_stack(ray: Ray, elements: Sequence[RefractiveElement]) -> Ray | None:
    """Return the ray leaving a stack of elements, or None if stopped.

    Elements are visited in geometric order (nearest entry hit first),
    independent of their order in the stack.
    """
    current: Ray | None = ray
    for _, result in traverse_stack(ray, elements):
        current = result.outgoing_ray
    return current


def trace_stack_path(
    ray: Ray, elements: Sequence[RefractiveElement], ray_length: float = 3.0
) -> "tuple[list[tuple[np.ndarray, np.ndarray, str]], Ray | None]":
    """Return ``(segments, final_ray)`` for rendering.

    Each segment is a ``(start, end, kind)`` tuple where ``kind`` is
    ``"incident"`` before the first element, ``"internal"`` inside an
    element, and ``"exit"`` after leaving one.

    Runs through the nopython stack kernel when every surface supports it,
    otherwise falls back to the Python traversal below.
    """
    ctx = build_trace_context(elements)
    if ctx is not None:
        return _trace_stack_path_fast(ray, ctx, ray_length, len(elements))
    segments: list[tuple[np.ndarray, np.ndarray, str]] = []
    current = ray
    kind = "incident"
    for _, result in traverse_stack(ray, elements):
        if result.incident_hit is not None:
            segments.append((current.origin, result.incident_hit.point, kind))
        for segment, endpoint in internal_rays(result):
            if endpoint is None:
                continue
            segments.append((segment.origin, endpoint, "internal"))
        if result.outgoing_ray is None:
            return segments, None
        current = result.outgoing_ray
        kind = "exit"
    end = current.origin + ray_length * current.direction
    segments.append((current.origin, end, kind))
    return segments, current
