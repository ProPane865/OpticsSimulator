"""Internal helpers shared across the :mod:`optics` package and ``rendering``."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import TypeVar

import numpy as np

_V = TypeVar("_V")


def cached_lru(
    cache: OrderedDict[tuple, _V],
    key: tuple,
    compute: Callable[[], _V],
    maxsize: int,
) -> _V:
    """Return ``cache[key]`` or insert ``compute()`` and evict to ``maxsize``.

    A hit is promoted to most-recently-used; a miss stores the fresh value
    (already the newest entry) and evicts the oldest once over capacity. Values
    are never ``None``, so a missing key is indistinguishable from ``None``.
    """
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached
    value = compute()
    cache[key] = value
    while len(cache) > maxsize:
        cache.popitem(last=False)
    return value


def coerce_ray_arrays(
    origins: np.ndarray, dirs: np.ndarray, powers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous ``float64`` ``(N, 3)`` origins/dirs and ``(N,)`` powers."""
    return (
        np.ascontiguousarray(origins, dtype=float),
        np.ascontiguousarray(dirs, dtype=float),
        np.ascontiguousarray(powers, dtype=float),
    )
