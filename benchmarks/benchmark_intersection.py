"""Ray/surface intersection benchmark: native kernel timings.

Scenario: one element's front surface, ``--rays`` rays (default 10,000)
through the production :meth:`optics.primitive.Surface.intersect` path (the
persistent native surface's seed grid + damped-Newton kernel), with setup
stages timed separately.

    python benchmarks/benchmark_intersection.py [--cold] [--json out.json]
"""

import argparse
import sys
import time
from pathlib import Path

import bench_common as bc


class NativeBench:
    def __init__(self, rays: int) -> None:
        self.rays = rays
        self.element = bc.make_element(0.0)
        self.surface = self.element.surface1
        origins, dirs = bc.make_rays(rays)
        self.ray_objects = bc.make_ray_objects(origins, dirs)

    def run(self) -> int:
        hits = 0
        intersect = self.surface.intersect
        for ray in self.ray_objects:
            hit = intersect(ray)
            hits += int(hit is not None)
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
        # One persistent native surface holds the seed grid + graph probe,
        # built once on first native use.
        _ = surface.native_surface

    seconds = bc.time_call(first_native_touch, warmup=0, repeats=1)
    report.add(
        "setup (one-time)", "native surface (seed grid + probe)", seconds
    )

    nb = NativeBench(args.rays)
    hits = nb.run()
    if hits == 0:
        print("warning: no rays hit the surface, check the ray fan", file=sys.stderr)
    nb_seconds = bc.time_call(nb.run, args.warmup, args.repeats, args.budget)
    report.add("timed", "Surface.intersect (per ray)", nb_seconds, args.rays, f"{hits:,} hits")

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
