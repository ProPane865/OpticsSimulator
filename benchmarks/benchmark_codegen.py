"""Stage-level profile of the AST -> Numba surface-codegen path.

``optics.numerical.codegen.compile_parametric_surface`` validates a surface's
``(x(u,v), y(u,v), z(u,v))`` expressions with :mod:`optics.numerical.expression`,
lowers them to scalar source, and emits a generated module with an
``njit``/``cfunc`` pair. Tangents are finite differences (no symbolic
differentiation), so the stages are: parse/validate/lower, emit the module
source, import it (which defines the njit dispatchers and cfuncs), force-compile
the kernels, and probe the cfunc pair.

Isolation model: every timing happens in a child process (``--child``), because
``compile_parametric_surface`` is ``lru_cache``'d and the generated module is
memoized in ``sys.modules`` and on disk -- a second in-process call is a no-op
and would report ~0s. ``--cold`` points ``NUMBA_CACHE_DIR`` at a throwaway temp
directory in the child so the ``cache=True`` kernels (``eval_local``,
``eval_grid``, ``derivatives_local``) actually compile; the ``cfunc`` pair is
never disk-cached, so it is a constant per-process floor in both modes.

    python benchmarks/benchmark_codegen.py [--cold] [--json out.json] [--surfaces ...]
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import bench_common as bc
import numpy as np

import optics.numerical.codegen as cg
from optics import constants as _c
from optics.numerical import expression
from optics.numerical.desc import probe_njit_pair

# Every surface shares one square parameter domain (matching the bundled
# meniscus in bench_common.element_config()).
DOMAIN = (-2.0, 2.0)

# (label, x_src, y_src, z_src). "front"/"rear" mirror the bundled meniscus;
# "hard" exercises a heavy scalar expression (the FD tangents cost the same as
# anywhere else -- they are finite differences of this eval); "div" exercises
# the hoisted ``_sdiv``/``_smod`` helpers (the div/mod rewriter); "null" is the
# inert surface padded trace contexts build via null_surface_cfuncs().
SURFACES: "dict[str, tuple[str, str, str]]" = {
    "front": ("u", "v", "-1 + sqrt(9 - u**2 - v**2)"),
    "rear": ("u", "v", "4.6 - sqrt(9 - u**2 - v**2)"),
    "hard": (
        "u",
        "v",
        "sin(u**2) + cos(u*v) + exp(-u**2 - v**2) * sqrt(9 - u**2 - v**2)",
    ),
    "div": ("u", "v", "u / (v - 0.5)"),
}
NULL_SCENARIO = "null"
ALL_SCENARIOS = list(SURFACES) + [NULL_SCENARIO]

# Child/parent line protocol.
_PREFIX = "OPTICSCG"


def _emit(kind: str, label: str, seconds: float) -> None:
    print(f"{_PREFIX}\t{kind}\t{label}\t{seconds:.6f}", flush=True)


def _emit_source(x_src: str, y_src: str, z_src: str) -> "tuple[str, str]":
    """Emit generated module source + cache key for a surface (mirrors _build_njit)."""
    u_min, u_max = DOMAIN
    v_min, v_max = DOMAIN
    codes = tuple(map(expression.scalar_source, (x_src, y_src, z_src)))
    source = cg._GENERATED_MODULE.format(
        x=codes[0],
        y=codes[1],
        z=codes[2],
        hu=cg.fd_step(u_min, u_max),
        hv=cg.fd_step(v_min, v_max),
        umin=repr(u_min),
        umax=repr(u_max),
        vmin=repr(v_min),
        vmax=repr(v_max),
        shrink=_c.FD_STEP_SHRINKS,
    )
    key = cg._codegen_key(x_src, y_src, z_src, u_min, u_max, v_min, v_max)
    return source, key


# --- child: the stages, timed single-shot on a cold process ------------------


def _child_stages(scenario: str) -> None:
    if scenario == NULL_SCENARIO:
        cg.null_surface_cfuncs.cache_clear()
        _child_stages_null()
        return

    x_src, y_src, z_src = SURFACES[scenario]
    total = 0.0

    def record(label: str, fn):
        nonlocal total
        t0 = time.perf_counter()
        value = fn()
        dt = time.perf_counter() - t0
        total += dt
        _emit("STAGE", label, dt)
        return value

    record(
        "scalar_source x3 (AST parse + validate + lower)",
        lambda: tuple(map(expression.scalar_source, (x_src, y_src, z_src))),
    )

    source, key = record("template format + _codegen_key", lambda: _emit_source(x_src, y_src, z_src))

    module = record(
        "_import_generated_module (write temp + exec_module -> define njit+cfunc)",
        lambda: cg._import_generated_module(key, source),
    )

    def _force() -> None:
        module.eval_local(0.7, 0.35)
        module.derivatives_local(0.7, 0.35)
        module.eval_grid(np.array([0.1, 0.2]), np.array([0.3, 0.4]))

    record("force-compile eval_local / derivatives_local / eval_grid", _force)

    record(
        "probe_njit_pair (compile eval/deriv cfunc pair)",
        lambda: probe_njit_pair(module.eval_cfn, module.deriv_cfn),
    )

    _emit("SUM", scenario, total)


def _child_stages_null() -> None:
    total = 0.0

    def timed(label: str, fn) -> None:
        nonlocal total
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        total += dt
        _emit("STAGE", label, dt)

    timed("null_surface_cfuncs() (inert padded-context cfuncs)", lambda: cg.null_surface_cfuncs())
    _emit("SUM", NULL_SCENARIO, total)


def _child_raw(scenario: str) -> None:
    if scenario == NULL_SCENARIO:
        cg.null_surface_cfuncs.cache_clear()
        t0 = time.perf_counter()
        cg.null_surface_cfuncs()
        _emit("RAW", scenario, time.perf_counter() - t0)
        return
    x_src, y_src, z_src = SURFACES[scenario]
    cg.compile_parametric_surface.cache_clear()
    t0 = time.perf_counter()
    cg.compile_parametric_surface(x_src, y_src, z_src, DOMAIN, DOMAIN)
    _emit("RAW", scenario, time.perf_counter() - t0)


def _child(mode: str, scenario: str) -> None:
    try:
        if mode == "stages":
            _child_stages(scenario)
        elif mode == "raw":
            _child_raw(scenario)
        else:
            raise ValueError(f"unknown child mode: {mode}")
    except Exception as exc:  # surface the failure to the parent via a line + stderr
        import traceback

        traceback.print_exc()
        _emit("ERROR", scenario, 0.0)
        raise SystemExit(f"child {mode} {scenario} failed: {exc}") from exc


# --- parent: spawn fresh interpreters and aggregate --------------------------


def _spawn(mode: str, scenario: str, cold: bool) -> "dict[str, float]":
    """Run one child; return ``{kind_or_label: seconds}`` (STAGE keys use label)."""
    env = os.environ.copy()
    cache_dir = tempfile.mkdtemp(prefix="optics-cg-cache-") if cold else None
    if cache_dir is not None:
        env["NUMBA_CACHE_DIR"] = cache_dir
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", mode, "--scenario", scenario]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
    finally:
        if cache_dir is not None:
            import shutil

            shutil.rmtree(cache_dir, ignore_errors=True)
    out: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        if not line.startswith(_PREFIX):
            continue
        _, kind, label, secs = line.split("\t")
        if kind == "ERROR":
            raise RuntimeError(f"child {mode} {scenario} reported ERROR:\n{proc.stderr}")
        if kind == "STAGE":
            out[f"S::{label}"] = float(secs)
        elif kind == "SUM":
            out["SUM"] = float(secs)
        elif kind == "RAW":
            out["RAW"] = float(secs)
    return out


def _measure(mode: str, scenario: str, cold: bool) -> "dict[str, float]":
    """Warm passes run twice and keep the second so the Numba cache is populated."""
    if cold:
        return _spawn(mode, scenario, cold=True)
    _spawn(mode, scenario, cold=False)  # prime the default Numba cache
    return _spawn(mode, scenario, cold=False)


def _top_stage(stages: "dict[str, float]") -> str:
    if not stages:
        return ""
    label = max(stages, key=lambda k: stages[k])
    return f"top: {label} ({stages[label] * 1000:.0f} ms)"


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage-level profile of optics.numerical.codegen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--surfaces",
        nargs="+",
        choices=ALL_SCENARIOS,
        default=None,
        help=f"surfaces to profile (default: all of {ALL_SCENARIOS})",
    )
    parser.add_argument(
        "--cold",
        action="store_true",
        help="additionally profile with a fresh NUMBA_CACHE_DIR (cold Numba compile)",
    )
    parser.add_argument("--json", type=Path, default=None, help="write report rows to JSON")
    parser.add_argument("--quiet", action="store_true", help="suppress the report table")
    # child-only plumbing
    parser.add_argument("--child", choices=["stages", "raw"], help=argparse.SUPPRESS)
    parser.add_argument("--scenario", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child:
        _child(args.child, args.scenario or "front")
        return 0

    scenarios = args.surfaces or ALL_SCENARIOS
    report = bc.Report("codegen path: per-stage cost (fresh interpreter)", args.quiet)

    passes = [("warm", False)] + ([("cold", True)] if args.cold else [])
    for pass_name, cold in passes:
        for scenario in scenarios:
            stages_raw = _measure("stages", scenario, cold)
            raw = _measure("raw", scenario, cold)
            stages = {k[3:]: v for k, v in stages_raw.items() if k.startswith("S::")}
            stage_sum = stages_raw.get("SUM", sum(stages.values()))
            raw_secs = raw.get("RAW", float("nan"))
            for label, secs in stages.items():
                pct = 100.0 * secs / stage_sum if stage_sum else 0.0
                report.add(f"{pass_name}: {scenario}", label, secs, note=f"{pct:4.0f}% of stages")
            report.add(
                f"{pass_name}: {scenario}",
                "stage sum",
                stage_sum,
                note=_top_stage(stages),
            )
            report.add(
                f"{pass_name}: {scenario}",
                "raw compile_parametric_surface",
                raw_secs,
                note=f"sum/raw = {stage_sum / raw_secs:.2f}x" if raw_secs and raw_secs == raw_secs else "",
            )

    report.print()
    if args.json:
        report.save_json(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
