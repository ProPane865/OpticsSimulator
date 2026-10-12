"""Parametric ray-tracing optics core."""

import warnings

try:
    from numba.core.errors import NumbaExperimentalFeatureWarning

    warnings.filterwarnings(
        "ignore",
        message=".*[Ff]irst-class function type.*",
        category=NumbaExperimentalFeatureWarning,
    )
except ImportError:  # pragma: no cover - numba is a hard dependency
    pass
