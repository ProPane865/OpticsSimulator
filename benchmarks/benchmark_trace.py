"""Single-element trace benchmark: Python reference vs native (serial/batched).

Scenario: one meniscus element, ``--rays`` rays (default 10,000), comparing
the pure-Python traversal (descriptors nulled so ``build_trace_context``
returns ``None``, conftest mechanism) against the public per-ray entry
point, the per-ray kernel with a prebuilt context, and the parallel batch
API (native threads).

    python benchmarks/benchmark_trace.py [--cold] [--json out.json]
"""

import argparse
import sys
import time
from pathlib import Path

import bench_common as bc

from optics.tracing import trace_segments_batch, trace_stack_path
from optics.tracing.context import _trace_stack_path_fast, build_trace_context


class TraceBench:
    def __init__(self, rays: int) -> None:
        self.rays = rays
        self.elements = bc.make_stack(1)
        self.origins, self.dirs = bc.make_rays(rays)
        self.ray_objs = bc.make_ray_objects(self.origins, self.dirs)
        self.ctx = build_trace_context(self.elements)
        if self.ctx is None:
            raise RuntimeError("element lost the native trace path")

    def python_reference(self) -> int:
        bc.force_python_path(self.elements)
        self.ctx = None
        return self._walk_python()

    def _walk_python(self) -> int:
        finals = 0
        for ray in self.ray_objs:
            # ReferenceSurface exposes no native descriptor, so build_trace_context
            # returns None and trace_stack_path falls through to the Python traversal.
            _, final = trace_stack_path(ray, self.elements, bc.RAY_LENGTH)
            finals += int(final is not None)
        return finals

    def per_ray_public(self) -> int:
        finals = 0
        for ray in self.ray_objs:
            _, final = trace_stack_path(ray, self.elements, bc.RAY_LENGTH)
            finals += int(final is not None)
        return finals

    def per_ray_fast(self) -> int:
        assert self.ctx is not None
        finals = 0
        for ray in self.ray_objs:
            _, final = _trace_stack_path_fast(ray, self.ctx, bc.RAY_LENGTH, len(self.elements))
            finals += int(final is not None)
        return finals

    def batch(self) -> int:
        out = trace_segments_batch(self.elements, self.origins, self.dirs, bc.RAY_LENGTH)
        assert out is not None
        return int(out[3].sum())


def run_scenario(name: str, rays: int) -> None:
    """One setup + one untimed pass; used by the cold-start subprocess."""
    bench = TraceBench(rays)
    if name == "batch":
        bench.batch()
    elif name == "fast":
        bench.per_ray_fast()
    else:
        raise SystemExit(f"unknown scenario: {name}")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    bc.add_common_args(parser, rays_default=10000)
    bc.add_cold_args(parser)
    parser.add_argument(
        "--ref-rays",
        type=int,
        default=1000,
        help="rays for the pure-Python reference (default: 1000)",
    )
    args = parser.parse_args(argv)

    if args.cold_child:
        t0 = time.perf_counter()
        run_scenario(args.scenario or "batch", args.rays)
        bc.finish_cold(t0)
        return 0

    report = bc.Report(f"trace: 1 element x {args.rays:,} rays", args.quiet)

    bench = TraceBench(args.rays)
    hits = bench.batch()
    if hits == 0:
        print("warning: no segments traced, check the ray fan", file=sys.stderr)

    fast_seconds = bc.time_call(bench.per_ray_fast, args.warmup, args.repeats, args.budget)
    report.add(
        "timed",
        "native per-ray (prebuilt ctx)",
        fast_seconds,
        args.rays,
        f"{hits / args.rays:.1f} segs/ray",
    )
    public_seconds = bc.time_call(bench.per_ray_public, args.warmup, args.repeats, args.budget)
    report.add(
        "timed",
        "native per-ray trace_stack_path",
        public_seconds,
        args.rays,
        f"rebuilds ctx per call: {public_seconds / fast_seconds:.1f}x prebuilt",
    )
    seconds = bc.time_call(bench.batch, args.warmup, args.repeats, args.budget)
    report.add("timed", "native batch trace_segments_batch", seconds, args.rays)

    ref = TraceBench(args.ref_rays)
    ref_finals = ref.python_reference()
    ref_seconds = bc.time_call(ref.python_reference, args.warmup, args.repeats, args.budget)
    ratio = (ref_seconds / args.ref_rays) / (fast_seconds / args.rays)
    report.add(
        "timed",
        "python reference (serial, ReferenceSurface)",
        ref_seconds,
        args.ref_rays,
        f"{ref_finals:,} exited; {ratio:,.0f}x slower per ray",
    )

    if args.cold:
        for scenario, label in (("batch", "batch kernel chain"), ("fast", "per-ray kernel chain")):
            cold, wall = bc.run_cold(Path(__file__), scenario, args.child_rays)
            report.add(
                "cold start (fresh interpreter)",
                f"first call, subprocess [{label}]",
                cold,
                note=f"first call only; full cold start incl. interpreter: {wall:.1f} s",
            )

    report.print()
    if args.json:
        report.save_json(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
