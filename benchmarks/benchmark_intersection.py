"""Ray/surface intersection benchmark: native kernel vs pure-Python oracle.

Scenario: one element's front surface, ``--rays`` rays (default 10,000).
The Python oracle path is the pure-Python ``ReferenceSurface`` ray/surface
intersection from ``tests/_reference.py`` (seed-grid + damped-Newton, no
native code), run on ``--ref-rays`` rays because the full count would take
minutes.

    python benchmarks/benchmark_intersection.py [--cold] [--json out.json]
"""

import argparse
import sys
import time
from pathlib import Path

import bench_common as bc

from optics.numerical.intersect import surface_intersect_njit
from optics.primitive import Ray


class NativeBench:
    def __init__(self, rays: int) -> None:
        self.rays = rays
        self.element = bc.make_element(0.0)
        self.surface = self.element.surface1
        self.desc = self.surface.njit_descriptor
        if self.desc is None:
            raise RuntimeError("surface lost the native kernel path")
        self.grid = self.surface.solver.njit_grid()
        self.origins, self.dirs = bc.make_rays(rays)

    def run(self) -> int:
        hits = 0
        for i in range(self.rays):
            _, _, _, ok = surface_intersect_njit(self.desc, self.grid, self.origins[i], self.dirs[i])
            hits += int(ok)
        return hits


class PythonOracleBench:
    def __init__(self, rays: int) -> None:
        self.rays = rays
        self.element = bc.make_element(0.0)
        bc.force_python_path([self.element])
        self.surface = self.element.surface1  # ReferenceSurface (pure Python)
        origins, dirs = bc.make_rays(rays)
        self.ray_objects = [Ray(origins[i], dirs[i], 1.0) for i in range(rays)]

    def run(self) -> int:
        hits = 0
        for ray in self.ray_objects:
            if self.surface.intersect(ray) is not None:
                hits += 1
        return hits


def run_scenario(name: str, rays: int) -> None:
    """One setup + one untimed pass; used by the cold-start subprocess."""
    if name == "native":
        NativeBench(rays).run()
    elif name == "python":
        PythonOracleBench(rays).run()
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
        help="rays for the pure-Python oracle (default: 1000)",
    )
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
    seconds = bc.time_call(lambda: surface.njit_descriptor, warmup=0, repeats=1)
    report.add("setup (one-time)", "native descriptor build", seconds)

    def rebuild_grid() -> None:
        surface.solver._grid_cache.clear()
        surface.solver._njit_grid_cache.clear()
        surface.solver.njit_grid()

    seconds = bc.time_call(rebuild_grid, warmup=0, repeats=1)
    report.add("setup (one-time)", "seed grid (njit_grid)", seconds)

    nb = NativeBench(args.rays)
    hits = nb.run()
    if hits == 0:
        print("warning: no rays hit the surface, check the ray fan", file=sys.stderr)
    nb_seconds = bc.time_call(nb.run, args.warmup, args.repeats, args.budget)
    report.add("timed", "native surface_intersect_njit", nb_seconds, args.rays, f"{hits:,} hits")

    po = PythonOracleBench(args.ref_rays)
    ref_hits = po.run()
    ref_seconds = bc.time_call(po.run, args.warmup, args.repeats, args.budget)
    ratio = (ref_seconds / args.ref_rays) / (nb_seconds / args.rays)
    report.add(
        "timed",
        "python ReferenceSurface.intersect",
        ref_seconds,
        args.ref_rays,
        f"{ref_hits:,} hits; {ratio:,.0f}x slower per ray",
    )

    if args.cold:
        for scenario, label in (("native", "native chain"), ("python", "python oracle (imports only)")):
            child_rays = args.child_rays if scenario == "native" else min(args.child_rays, 100)
            cold, wall = bc.run_cold(Path(__file__), scenario, child_rays)
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
