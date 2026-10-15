import numpy as np
import pytest

import optics.tracing as rf
from optics.measurement_plane import MeasurementPlane
from optics.primitive import Ray
from optics.tracing import stack as stack_mod
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


def test_measure_parallel_matches_serial(sphere):
    rays = _rays()
    parallel = _plane().measure(rays, [sphere])
    serial = _plane()._measure_element_stack(rays, [sphere])
    assert parallel == pytest.approx(serial, rel=1e-12, abs=1e-12)


def test_measure_is_deterministic(sphere):
    rays = _rays()
    first = _plane().measure(rays, [sphere])
    second = _plane().measure(rays, [sphere])
    assert first == pytest.approx(second, rel=0.0, abs=0.0)


def test_trace_stack_path_fast_matches_serial(sphere, monkeypatch):
    ray = Ray(np.array([0.2, 0.1, 3.0]), np.array([0.0, 0.0, -1.0]), 1.0)
    fast_segs, fast_final = rf.trace_stack_path(ray, [sphere])
    monkeypatch.setattr(stack_mod, "build_trace_context", lambda elements: None)
    slow_segs, slow_final = rf.trace_stack_path(ray, [sphere])
    assert len(fast_segs) == len(slow_segs)
    for (a0, b0, k0), (a1, b1, k1) in zip(fast_segs, slow_segs):
        assert k0 == k1
        np.testing.assert_allclose(a0, a1, atol=1e-9)
        np.testing.assert_allclose(b0, b1, atol=1e-9)
    if fast_final is None or slow_final is None:
        assert fast_final is slow_final
    else:
        np.testing.assert_allclose(fast_final.origin, slow_final.origin, atol=1e-9)
        np.testing.assert_allclose(
            fast_final.direction, slow_final.direction, atol=1e-9
        )


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
