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

- `optics/`: numerics core (no GUI/OpenGL dependencies):
  - `surface.py`: parametric surfaces compiled from SymPy expressions
  - `solver.py`: damped-Newton ray/surface intersection and parameter inversion
  - `refraction.py`: refraction/reflection, closed-element tracing, stack traversal
  - `sidewall.py`: vectorized Möller–Trumbore intersection for lens rims
  - `measurement_plane.py`: planar detector accumulating ray power
  - `source.py`, `ray.py`, `aperture.py`, `material.py`, `transform.py`, `vector.py`
  - `config.py`: JSON → element/stack loading
  - `project.py`: versioned project documents (elements + source + measurement planes)
  - `examples/`: bundled example stacks (package data)
- `rendering/`: vispy meshing (`meshing.py`), scene building (`visualize.py`), and a
  standalone demo (`demo.py`)
- `ui/`: PySide6 desktop application (`app.py`, `main.ui`)

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
