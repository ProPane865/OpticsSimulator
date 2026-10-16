"""Stage-level profile of the SymPy -> Numba codegen path.

``optics.numerical.codegen.compile_parametric_surface`` turns a surface's
``(x(u,v), y(u,v), z(u,v))`` expressions into (a) a lambdified NumPy fallback
and (b) a generated module with a ``njit``/``cfunc`` pair. This script times
each stage of that path on a *fresh* interpreter so the numbers reflect a real
first call, then compares the stage sum against the raw function cost.

Isolation model: every timing happens in a child process (``--child``), because
``compile_parametric_surface`` is ``lru_cache``'d and the generated module is
memoized in ``sys.modules`` and on disk -- a second in-process call is a no-op
and would report ~0s. ``--cold`` points ``NUMBA_CACHE_DIR`` at a throwaway temp
directory in the child so the ``cache=True`` kernels (``eval_local``,
``derivatives_local``, the descriptor probe) actually compile; the ``cfunc``
pair is never disk-cached, so it is a constant per-process floor in both modes.

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
import sympy as sp

import optics.numerical.codegen as cg
from optics.numerical.desc import probe_njit_pair

# (label, x_src, y_src, z_src). "front"/"rear" mirror the bundled meniscus in
# bench_common.element_config(); "hard" stresses sympy.diff; "null" is the inert
# surface padded trace contexts build via null_surface_cfuncs().
SURFACES: "dict[str, tuple[str, str, str]]" = {
    "front": ("u", "v", "-1 + sqrt(9 - u**2 - v**2)"),
    "rear": ("u", "v", "4.6 - sqrt(9 - u**2 - v**2)"),
    "hard": (
        "u",
        "v",
        "sin(u**2) + cos(u*v) + exp(-u**2 - v**2) * sqrt(9 - u**2 - v**2)",
    ),
    # Exercises the hoisted ``_sdiv`` helper (the div/mod rewriter). ``Mod`` is
    # excluded because sympy cannot differentiate it for the lambdify fallback.
    "div": ("u", "v", "u / (v - 0.5)"),
}
NULL_SCENARIO = "null"
ALL_SCENARIOS = list(SURFACES) + [NULL_SCENARIO]

# Child/parent line protocol.
_PREFIX = "OPTICSCG"


def _emit(kind: str, label: str, seconds: float) -> None:
    print(f"{_PREFIX}\t{kind}\t{label}\t{seconds:.6f}", flush=True)


# --- child: the stages, timed single-shot on a cold process ------------------


def _child_stages(scenario: str) -> None:
    u, v = sp.symbols("u v")

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

    parsed = record("parse_expr x3 (SymPy)", lambda: tuple(map(sp.parse_expr, (x_src, y_src, z_src))))
    x_expr, y_expr, z_expr = parsed

    diffed = record(
        "sympy.diff x6 (partial tangents)",
        lambda: (
            sp.diff(x_expr, u),
            sp.diff(y_expr, u),
            sp.diff(z_expr, u),
            sp.diff(x_expr, v),
            sp.diff(y_expr, v),
            sp.diff(z_expr, v),
        ),
    )
    xu_expr, yu_expr, zu_expr = diffed[0], diffed[1], diffed[2]
    xv_expr, yv_expr, zv_expr = diffed[3], diffed[4], diffed[5]

    codes = record(
        "_scalar_code x9 (pycode + AST div-rewrite + unparse)",
        lambda: tuple(
            cg._scalar_code(e) for e in (x_expr, y_expr, z_expr, xu_expr, yu_expr, zu_expr, xv_expr, yv_expr, zv_expr)
        ),
    )

    def _fmt() -> "tuple[str, str]":
        source = cg._GENERATED_MODULE.format(
            x=codes[0],
            y=codes[1],
            z=codes[2],
            xu=codes[3],
            yu=codes[4],
            zu=codes[5],
            xv=codes[6],
            yv=codes[7],
            zv=codes[8],
        )
        return source, cg._codegen_key(x_src, y_src, z_src)

    source, key = record("template format + _codegen_key", _fmt)

    module = record(
        "_import_generated_module (write temp + exec_module -> njit+cfunc compile)",
        lambda: cg._import_generated_module(key, source),
    )

    def _force() -> None:
        module.eval_local(0.7, 0.35)
        module.derivatives_local(0.7, 0.35)

    record("force-compile eval_local / derivatives_local", _force)

    record(
        "probe_njit_pair (compile _probe_pair njit)",
        lambda: probe_njit_pair(module.call_cfn),
    )

    def _lambdify() -> int:
        n = 0
        for expr in (x_expr, y_expr, z_expr, xu_expr, yu_expr, zu_expr, xv_expr, yv_expr, zv_expr):
            sp.lambdify((u, v), expr, modules=["numpy"])
            n += 1
        return n

    record("sp.lambdify x9 (NumPy fallback path)", _lambdify)

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
    cg.compile_parametric_surface(x_src, y_src, z_src)
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
