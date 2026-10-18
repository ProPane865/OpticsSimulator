"""Ray/surface intersection benchmark: native kernel timings.

Scenario: one element's front surface, ``--rays`` rays (default 10,000)
through the native ``surface_intersect_kernel`` kernel (seed grid +
damped-Newton), with setup stages timed separately.

    python benchmarks/benchmark_intersection.py [--cold] [--json out.json]
"""

import argparse
import sys
import time
from pathlib import Path

import bench_common as bc

from optics.numerical.intersect import surface_intersect_kernel


class NativeBench:
    def __init__(self, rays: int) -> None:
        self.rays = rays
        self.element = bc.make_element(0.0)
        self.surface = self.element.surface1
        self.desc = self.surface.native_descriptor
        if self.desc is None:
            raise RuntimeError("surface lost the native kernel path")
        self.grid = self.surface.solver.native_grid()
        self.origins, self.dirs = bc.make_rays(rays)

    def run(self) -> int:
        hits = 0
        for i in range(self.rays):
            _, _, _, ok = surface_intersect_kernel(self.desc, self.grid, self.origins[i], self.dirs[i])
            hits += int(ok)
        return hits


def run_scenario(name: str, rays: int) -> None:
    """One setup + one untimed pass; used by the cold-start subprocess."""
    if name == "native":
        NativeBench(rays).run()
    else:
        raise SystemExit(f"unknown scenario: {name}")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    bc.add_common_args(parser, rays_default=10000)
    bc.add_cold_args(parser)
    args = parser.parse_args(argv)

    if args.cold_child:
        t0 = time.perf_counter()
        run_scenario(args.scenario or "native", args.rays)
        bc.finish_cold(t0)
        return 0

    report = bc.Report(f"intersection: 1 surface x {args.rays:,} rays", args.quiet)

    element = bc.make_element(0.0)
    surface = element.surface1
    seconds = bc.time_call(lambda: bc.make_element(0.0), warmup=0, repeats=1)
    report.add("setup (one-time)", "element build incl. expression codegen", seconds)

    def first_native_touch() -> None:
        # One persistent native surface holds descriptor + seed grid + graph
        # probe, built once on first native use.
        _ = surface.native_surface

    seconds = bc.time_call(first_native_touch, warmup=0, repeats=1)
    report.add(
        "setup (one-time)", "native surface (descriptor + seed grid + probe)", seconds
    )

    nb = NativeBench(args.rays)
    hits = nb.run()
    if hits == 0:
        print("warning: no rays hit the surface, check the ray fan", file=sys.stderr)
    nb_seconds = bc.time_call(nb.run, args.warmup, args.repeats, args.budget)
    report.add("timed", "native surface_intersect_kernel", nb_seconds, args.rays, f"{hits:,} hits")

    if args.cold:
        cold, wall = bc.run_cold(Path(__file__), "native", args.child_rays)
        report.add(
            "cold start (fresh interpreter)",
            "first call, subprocess [native chain]",
            cold,
            note=f"first call only; full cold start incl. interpreter: {wall:.1f} s",
        )

    report.print()
    if args.json:
        report.save_json(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
