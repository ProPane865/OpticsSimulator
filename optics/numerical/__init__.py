"""Numerical kernels and solvers: ray/surface intersection, parameter
inversion, plane and mesh intersection, vector helpers.

The hot paths run in the compiled native module (``optics._native``); this
package is a dependency-free leaf layer of :mod:`optics`: it only depends on
:mod:`optics.constants`, never on :mod:`optics.primitive`. Import from the
specific submodule (``optics.numerical.vector``, ``optics.numerical.solver``,
...).
"""
