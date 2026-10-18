# OpticsSimulator

An interactive, parametric ray-tracing optics simulator. Define lenses as
parametric surfaces, trace rays through refractive stacks (including total
internal reflection and sidewalls), and visualize the result and measurement
planes in a 3D viewport.

## Installation

Requires Python 3.10+ and a C++17 compiler: installing the package compiles
the native kernel extension (`optics._native`, pybind11 via scikit-build-core
and CMake ≥ 3.24; `cmake`/`ninja` are pulled from PyPI into the build
environment when not already present). Use the MSVC Build Tools or MinGW-w64
on Windows, and gcc or clang on Linux/macOS.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -e .[dev]
```

On Linux, the offscreen Qt/vispy backend used by the tests needs system GL/XCB
libraries (see `.github/workflows/ci.yml` for the exact package list).

## Running the app

```bash
opticssim
# or
python -m ui.app
```

The window loads the bundled `optics/examples/doublet.json` by default
(`triplet.json` is bundled as well). Use **File -> Open** to load another
stack and **File -> Save As** to write one back.

To view the example stack without the full UI:

```bash
python -m rendering.demo
```

## Architecture

Hot numerics (expression evaluation, Newton solvers, intersection, stack
tracing, measurement) run in a compiled C++ kernel library exposed to Python
as `optics._native` (type stubs in `optics/_native.pyi`). NumPy is used for
array buffers only.

- `native/`: C++17 kernels (built as the `optics_core` object library and the
  `_native` pybind11 module), layered `common ← expression ← geometry ←
  intersection ← tracing ← measurement`, with headers under
  `include/optics/<module>/` and sources in per-module directories:
  - `expression/`: parser/validator for the restricted surface-expression
    grammar plus an interpreter over the lowered expression tree
  - `geometry/`: surface models and descriptors — expression evaluation,
    finite-difference tangents, aperture membership and world-space normals
  - `intersection/`: seed-grid accelerated damped-Newton ray/surface
    intersection, parameter inversion and the plane/mesh primitives
  - `tracing/`: element contexts, refraction/total internal reflection,
    the element/stack walk and the shared single-trace hop recording
  - `measurement/`: plane power accumulation, direct/stacked plane
    measurement and the fused trace+planes batch pass
  - `common/`: vector/matrix math and small-system linalg (`rotation_from_z`
    incl.), parallel dispatch (native threading), `constants.hpp`;
    `-ffp-contract=off` keeps floating-point evaluation order faithful to
    strict IEEE semantics
  - `tests/`: native unit test TUs built into one ctest binary (see Testing)
- `optics/`: numerics core (NumPy buffers + native kernels, no GUI/OpenGL
  dependencies):
  - `primitive/`: geometric primitives — `ray.py`, `transform.py`, `hit.py`,
    `material.py`, `aperture.py`, `sidewall.py` (lens rims intersected via
    the native Möller–Trumbore mesh kernel) and `surface.py` (parametric
    surfaces compiled from validated expression strings)
  - `numerical/`: thin Python wrappers around the native kernels —
    `codegen.py` (restricted, safe surface-expression grammar enforced by the
    native parser; expression strings → native surface model via
    `_native.compile_surface`, lru-cached), `_kernels.py` and `desc.py`
    (descriptor/context assembly), `solver.py`, `intersect.py`, `invert.py`,
    `plane.py`, `mesh.py` (native intersection/inversion/plane/mesh kernels)
    and `vector.py`
  - `tracing/`: element/stack tracing on the native kernels —
    `element.py` (`RefractiveElement`: cached sidewall and element-bounds
    construction feeding the native trace contexts), `kernels.py` (native
    trace-kernel entry points), `context.py` (opaque,
    element-count-independent native trace stacks, batched parallel tracing,
    and the fused segments+planes pass), `stack.py` (rendering path building)
  - `measurement_plane.py`: planar detector accumulating ray power (native
    accumulation, fused with tracing when a stack is traced)
  - `constants.py`: numerical tolerances and solver defaults
  - `source.py`: point sources and ray emission
  - `config.py`: JSON → element/stack loading
  - `project.py`: versioned project documents (elements + source + measurement planes)
  - `examples/`: bundled example stacks (`doublet.json`, `triplet.json`)
- `rendering/`: vispy meshing (`meshing.py`, with a geometry-signature cache
  so editing a lens reuses identical local meshes), scene building
  (`visualize.py`), and a standalone demo (`demo.py`)
- `ui/`: PySide6 desktop application (`main.ui` is the Qt Designer layout):
  - `app.py`: composition root — `OpticsMainWindow`, project load/save and
    menu dialogs, plus `make_display`/`main` entry points; a unified refresh
    runs one combined native trace pass serving both the rendered rays and
    every measurement plane
  - `model.py`: Qt-free `ProjectState` (element/source/plane state, unified
    naming, edit classification for the fast lens-edit path, and
    versioned-document load/save via `optics.project`)
  - `scene.py`: `SceneController` — vispy view + node bookkeeping for elements,
    rays, the source marker and measurement planes
  - `editors.py`: `ElementEditor`/`SourceEditor`/`PlaneEditor` controllers over
    the dock widgets (populate/apply/add/remove)
  - `widgets.py`: shared PySide6 helpers (`find_child`, `configure_spin`,
    `enable_widgets`, `run_validated`)

## Project file format

**File -> Save As** writes a versioned project document that also stores the
light source and measurement planes:

```json
{
  "version": 1,
  "elements": {
    "lens1": { ... }
  },
  "source": {
    "position": [0, 0, 3],
    "power": 1.0, "num_rays": 16,
    "source_type": "single_point",
    "cone_angle": 60.0
  },
  "measurement_planes": [
    {
      "name": "plane1",
      "origin": [0, 0, -3],
      "normal": [0, 0, 1],
      "u_range": [-2, 2],
      "v_range": [-2, 2],
      "resolution": [64, 64]
    }
  ]
}
```

## Optical stack format

A stack is a JSON object mapping element names to element configs:

```json
{
  "lens1": {
    "surface1": {
      "x": "u",
      "y": "v",
      "z": "-1 + sqrt(9 - u**2 - v**2)",
      "u_range": [-2, 2],
      "v_range": [-2, 2],
      "aperture": {
        "radius": 0.75
      }
    },
    "surface2": {
      "x": "u",
      "y": "v",
      "z": "4.6 - sqrt(9 - u**2 - v**2)",
      "u_range": [-2, 2],
      "v_range": [-2, 2],
      "aperture": {
        "radius": 0.75
      }
    },
    "material": {
      "refractive_index": 1.5
    },
    "position": [0.0, 0.0, 0.0],
    "orientation": [0.0, 0.0, 1.0]
  }
}
```

Apertures are optional and circular: `radius` is required, `center`
(`[cx, cy]`, default `(0, 0)`) is optional, and `type` defaults to
`"circular"`.

### Surface expressions

Each surface maps parameters `(u, v)` over `u_range`/`v_range` to a local point
`(x(u, v), y(u, v), z(u, v))`. Expressions use a restricted, safe arithmetic
language (parsed, validated, and compiled by the native C++ parser —
no SymPy, and no `eval` of raw strings):

- Variables: `u`, `v`
- Literals: integers and floats
- Operators: `+ - * / ** %` and unary `+`/`-`
- Functions: `sqrt`, `sin`, `cos`, `tan`, `exp`, `log`, `abs`

Division and modulo follow NumPy scalar semantics (`inf`/`nan`, never a
`ZeroDivisionError`). Anything outside this grammar — attribute access,
subscripts, arbitrary calls, imports, `pi`/`e` — is rejected when the stack
loads. Surface tangents (and therefore normals) are computed with finite
differences rather than symbolic differentiation, with a domain-aware policy
implemented in the native kernels.

## Testing

```bash
python -m pytest
```

Lint and type checks (the commands CI runs):

```bash
ruff check .
mypy optics rendering ui
```

Native C++ tests:

```bash
cmake -S . -B build-native -DOPTICS_BUILD_MODULE=OFF -DOPTICS_BUILD_TESTS=ON
cmake --build build-native --config Release --target optics_native_tests
ctest --test-dir build-native -C Release --output-on-failure
```

On headless machines set `QT_QPA_PLATFORM=offscreen` for the UI tests (as CI
does).

## Benchmarks

Standalone perf scripts (no extra dependencies) used to judge architecture
changes against real numbers; run them from the repo root:

```bash
python benchmarks/benchmark_intersection.py  # 1 surface x 10k rays, native kernel, setup stages broken out
python benchmarks/benchmark_trace.py         # 1 element x 10k rays: native per-ray, prebuilt ctx, native batch
python benchmarks/benchmark_stack.py         # 2 elem x 10k and 4 elem x 100k: batch tracing (native threads), cold vs warm
python benchmarks/benchmark_codegen.py       # stage-level profile of surface compilation (C++ parse + first eval passes)
python benchmarks/profile_lens_edit.py       # lens create/modify hot path: per-phase breakdown, scaling matrix, cProfile
```

The three tracing benchmarks (`intersection`, `trace`, `stack`) print a
per-scenario table with rays/sec and ns/ray; `profile_lens_edit.py` reports
per-phase times plus a unified-apply scaling matrix over element/plane/ray
counts. All four share `--rays/--repeats/--warmup/--budget` (the repeat
budget caps cumulative timed seconds per scenario), `--json out.json` for
machine-readable output, and `--cold` to additionally re-run itself in a
fresh subprocess interpreter; the native kernels carry no JIT cache, so cold
runs are dominated by interpreter start-up and first-call setup.
`benchmark_stack.py` adds `--two-rays`/`--four-rays` (both stack sizes share
one native code path, so there is nothing to re-compile between sizes); and
`profile_lens_edit.py` adds `--elements/--planes/--mesh-n/--no-cprofile`.
`benchmark_codegen.py` times compilation stages in isolated child processes
and takes `--surfaces` (a scenario list), `--json` and `--quiet`. Setup costs
(expression compilation, seed grids, context build) are timed separately from
the ray loops; ray fans are deterministically seeded.
