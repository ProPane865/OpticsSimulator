"""Compile parametric surface expressions to native kernels (no SymPy).

A :class:`~optics.primitive.surface.Surface`'s ``(x(u, v), y(u, v), z(u, v))``
expressions are parsed and validated by the C++ parser in
``optics._native.compile_surface``, which owns the restricted surface
expression grammar (variables ``u``/``v``, numeric literals, the arithmetic
operators, and the scalar functions ``sqrt``, ``sin``, ``cos``, ``tan``,
``exp``, ``log``, ``abs``) and lowers to an expression tree the native
kernels evaluate directly with NumPy scalar semantics (``inf``/``nan``-style
division and modulo).

Because the supported grammar always lowers to evaluatable code, a compiled
surface is guaranteed to build: anything outside the grammar is rejected at
parse time with :class:`ExpressionError`.
"""

import functools
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

from . import _native


class ExpressionError(ValueError):
    """Raised for any surface expression outside the supported language."""


class SurfaceFunctions(NamedTuple):
    """Everything a compiled parametric surface needs, in named fields."""

    x_src: str
    y_src: str
    z_src: str
    eval_local: "Callable[[float, float], np.ndarray]"
    eval_grid: "Callable[[np.ndarray, np.ndarray], np.ndarray]"
    derivatives_local: "Callable[[float, float], tuple[np.ndarray, np.ndarray]]"
    model: _native.SurfaceModel


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
