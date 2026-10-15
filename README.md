# OpticsSimulator

An interactive, parametric ray-tracing optics simulator. Define lenses as
parametric surfaces, trace rays through refractive stacks (including total
internal reflection and sidewalls), and visualize the result and measurement
planes in a 3D viewport.

## Installation

Requires Python 3.10+.

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

The window loads the bundled `optics/examples/doublet.json` by default. Use
**File -> Open** to load another stack and **File -> Save As** to write one back.

To view the example stack without the full UI:

```bash
python -m rendering.demo
```

## Architecture

- `optics/`: numerics core (NumPy/Numba, no GUI/OpenGL dependencies):
  - `primitive/`: geometric primitives — `ray.py`, `transform.py`, `hit.py`,
    `material.py`, `aperture.py`, `sidewall.py` (vectorized Möller–Trumbore
    lens rims) and `surface.py` (parametric surfaces compiled from SymPy
    expressions)
  - `numerical/`: solvers and Numba kernels — `codegen.py` (SymPy → Numba
    expression codegen), `solver.py` (damped-Newton ray/surface intersection
    and parameter inversion), `intersect.py`, `invert.py`, `newton.py`,
    `desc.py` (descriptor access kernels), `_kernels.py` (flat-tuple data
    layouts shared with the nopython kernels), `mesh.py` (ray/mesh kernel),
    `plane.py` (Numba ray/plane intersection kernel), `linalg.py`,
    `vector.py`, `parallel.py` (chunked GIL-free dispatch)
  - `tracing/`: refraction/reflection and element/stack tracing —
    `direction.py` (refraction/total-internal-reflection direction kernels),
    `element.py` (`RefractiveElement`: containment, face hits, closed-element
    Python tracing), `kernels.py` (nopython element/stack walk),
    `context.py` (flattened, capacity-padded stack contexts and batched
    parallel tracing), `stack.py` (geometric-order stack traversal and
    rendering path building)
  - `measurement_plane.py`: planar detector accumulating ray power
  - `constants.py`: numerical tolerances and solver defaults
  - `source.py`: point sources and ray emission
  - `config.py`: JSON → element/stack loading
  - `project.py`: versioned project documents (elements + source + measurement planes)
  - `examples/`: bundled example stacks (package data)
- `rendering/`: vispy meshing (`meshing.py`), scene building (`visualize.py`), and a
  standalone demo (`demo.py`)
- `ui/`: PySide6 desktop application (`main.ui` is the Qt Designer layout):
  - `app.py`: composition root — `OpticsMainWindow`, project load/save and menu
    dialogs, plus `make_display`/`main` entry points
  - `model.py`: Qt-free `ProjectState` (element/source/plane state, unified
    naming, and versioned-document load/save via `optics.project`)
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
      "u_range": [-2, 2], "v_range": [-2, 2],
      "aperture": { "radius": 0.75 }
    },
    "material": {
      "refractive_index": 1.5
    },
    "position": [0.0, 0.0, 0.0],
    "orientation": [0.0, 0.0, 1.0]
  }
}
```

## Testing

```bash
python -m pytest
```

## Benchmarks

Standalone perf scripts (no extra dependencies) used to judge architecture
changes against real numbers; run them from the repo root:

```bash
python benchmarks/benchmark_intersection.py   # 1 surface x 10k rays, njit vs Python oracle
python benchmarks/benchmark_trace.py          # 1 element x 10k rays: Python ref, per-ray, batch x threads
python benchmarks/benchmark_stack.py          # 2 elem x 10k and 4 elem x 100k: batch, threads, cold/warm JIT
```

Each prints a per-scenario table with rays/sec and ns/ray and supports
`--rays/--repeats/--budget` (the repeat budget caps cumulative timed seconds
per scenario), `--ref-rays` for the slow pure-Python reference rows,
`--json out.json` for machine-readable output, and `--cold` to additionally
re-run itself in a subprocess with `NUMBA_CACHE_DIR` pointed at a fresh temp
directory — measuring the first call against a fully cold Numba cache
(the padded-capacity-4 stack kernel chain compiles for minutes; note the
2- and 4-element stacks share one compiled signature, so one cold JIT
serves both). Setup costs (SymPy codegen, seed grid, context build) are
timed separately from the ray loops; ray fans are deterministically seeded.
The 4-element x 100k single-thread scenario is minutes by construction (the
repeat budget never cuts the first repeat); use `--four-rays` to scale it.
