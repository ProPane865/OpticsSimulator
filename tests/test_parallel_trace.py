import numpy as np
import pytest

import optics.tracing as rf
from optics.measurement_plane import MeasurementPlane
from optics.primitive import Ray
from optics.tracing.context import _SEGMENT_KIND_NAMES


def _rays(n=24, seed=3):
    rng = np.random.default_rng(seed)
    rays = []
    for _ in range(n):
        x, y = rng.uniform(-0.6, 0.6, 2)
        d = rng.normal(size=3) * 0.05
        d[2] = -1.0
        d = d / np.linalg.norm(d)
        rays.append(Ray(np.array([x, y, 3.0]), d, 1.0))
    return rays


def _plane(resolution=(6, 6)):
    return MeasurementPlane(
        origin=np.array([0.0, 0.0, -3.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        u_range=(-2.0, 2.0),
        v_range=(-2.0, 2.0),
        resolution=resolution,
    )


def test_measure_is_deterministic(sphere):
    rays = _rays()
    first = _plane().measure(rays, [sphere])
    second = _plane().measure(rays, [sphere])
    assert first == pytest.approx(second, rel=0.0, abs=0.0)


def test_trace_segments_batch_matches_trace_stack_path(sphere):
    rays = _rays()
    origins = np.array([r.origin for r in rays], dtype=float)
    dirs = np.array([r.direction for r in rays], dtype=float)
    out = rf.trace_segments_batch([sphere], origins, dirs, 3.0)
    assert out is not None
    starts, ends, kinds, counts = out
    for i, ray in enumerate(rays):
        segs, _ = rf.trace_stack_path(ray, [sphere], ray_length=3.0)
        assert counts[i] == len(segs)
        for j, (a, b, kind) in enumerate(segs):
            np.testing.assert_allclose(starts[i, j], a, atol=1e-9)
            np.testing.assert_allclose(ends[i, j], b, atol=1e-9)
            assert _SEGMENT_KIND_NAMES[kinds[i, j]] == kind


def test_empty_stack_is_not_accelerated():
    assert rf.build_trace_context([]) is None
    ray = Ray(np.array([0.1, 0.2, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)
    segs, final = rf.trace_stack_path(ray, [])
    assert [k for _, _, k in segs] == ["incident"]
    assert final is not None
