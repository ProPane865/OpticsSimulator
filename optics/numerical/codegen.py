"""Compile parametric surface expressions to native kernels (no SymPy).

A :class:`~optics.primitive.surface.Surface`'s ``(x(u, v), y(u, v), z(u, v))``
expressions are parsed and validated by the C++ parser in
``optics._native.compile_surface`` (the grammar mirrors
:mod:`optics.numerical.expression`) and lowered to an expression tree that the
native kernels evaluate directly. The Python reference oracle
(:mod:`optics.numerical.expression` plus ``tests/_reference.py``) keeps an
independent implementation of the same language and scalar semantics
(``inf``/``nan``-style division and modulo, domain-aware finite differences).

Because the supported grammar always lowers to evaluatable code, a compiled
surface is guaranteed to build: anything outside the grammar is rejected at
parse time with :class:`~optics.numerical.expression.ExpressionError`.
"""

import functools
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from .. import _native
from .expression import ExpressionError


class SurfaceFunctions(NamedTuple):
    """Everything a compiled parametric surface needs, in named fields."""

    x_src: str
    y_src: str
    z_src: str
    eval_local: Any
    eval_grid: Any
    derivatives_local: Any
    model: Any


@functools.lru_cache(maxsize=64)
def compile_parametric_surface(
    x_src: str,
    y_src: str,
    z_src: str,
    u_range: Sequence[float],
    v_range: Sequence[float],
) -> SurfaceFunctions:
    """Validate and compile ``(x, y, z)`` into a native surface model."""
    u_min, u_max = float(u_range[0]), float(u_range[1])
    v_min, v_max = float(v_range[0]), float(v_range[1])
    try:
        model = _native.compile_surface(
            str(x_src), str(y_src), str(z_src), u_min, u_max, v_min, v_max
        )
    except ValueError as exc:
        raise ExpressionError(str(exc)) from exc

    def derivatives_local(u: float, v: float) -> "tuple[np.ndarray, np.ndarray]":
        ru, rv = model.derivatives_local(float(u), float(v))
        return ru, rv

    return SurfaceFunctions(
        x_src=x_src,
        y_src=y_src,
        z_src=z_src,
        eval_local=model.eval_local,
        eval_grid=model.eval_grid,
        derivatives_local=derivatives_local,
        model=model,
    )
