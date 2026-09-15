# Optics Simulator

A 3D **geometric optics** simulator: ray tracing through arbitrary lens
surfaces, followed by an **irradiance map** on a detector plane built from the
ray-map Jacobian (flux conservation). It ships a headless physics core and a
PySide6 + OpenGL interactive GUI.

## Physics

- Surfaces are described by a sag `z = g(x, y)`. The surface normal comes from
  the gradient: `n ∝ (-∂g/∂x, -∂g/∂y, 1)`.
- **Refraction** uses the vector form of Snell's law:
  `d' = η d + (η cos_i − cos_t) n`, with `η = n₁/n₂`. When `η²(1 − cos_i²) > 1`
  the ray total-internal-reflects.
- **Aperture clipping**: circular surfaces carry an aperture radius; rays whose
  hit point lies outside it are marked *blocked* and excluded from the map.
- **Irradiance mapping** follows from radiometric flux conservation through a
  ray bundle: `I_out = I_in / |det J|`, where `J` is the Jacobian of the map
  from source-plane coordinates to detector-plane coordinates (obtained by
  finite differences of the traced ray field). Near caustics `|det J| → 0`;
  those cells are clamped and masked.

Ray tracing is a single **Numba** kernel (`optics/accel.py`) that re-implements
each surface's geometry from its descriptor, so Numba is a required dependency.

## Layout

```
optics/            headless physics core
  accel.py         Numba ray-tracing kernel (single backend)
  surfaces.py      Plane / ConicSurface / PolySagSurface
  rays.py          RayStack (batched ray tracing)
  stack.py         OpticalStack + TraceResult
  snell.py         vector Snell's law / TIR
  irradiance.py    Jacobian map, flux conservation
  source.py        IrradianceSource (2D irradiance pattern -> rays)
gui/               PySide6 + OpenGL interactive viewer
  app.py           main window + panels wiring
  viewport.py      OpenGL render of surfaces, rays, source/detector maps
  models.py        Scene model + presets (bi_convex, thin_lens, ball_lens, collimator)
  panels.py        source / surface / detector control panels
examples/          runnable demos (see below)
tests/             pytest suite
```

## Installation

```bash
pip install -e .          # runtime deps: numpy, scipy, numba, PySide6, PyOpenGL
pip install -e ".[dev]"   # + pytest
```

## Run the GUI

```bash
optics-simulator            # console entry point
# or
python -m gui
```

Presets live under **System** in the menu. Use the Source panel to change the
emission pattern (Gaussian / disk / uniform), the Surface panel to add/remove
surfaces and edit radius/conicity/aperture, and **Auto-focus** to place the
detector at the best-focus plane. Orbit with the left mouse button, pan with
the middle/right button, zoom with the wheel.

## Use the core library

```python
import numpy as np
from optics import ConicSurface, OpticalStack, IrradianceSource, irradiance_from_trace

stack = OpticalStack("bi_convex")
stack.add_surface(ConicSurface("front", radius=50.0, conicity=0.0, z0=0.0,
                               aperture=45.0), medium_index=1.5)
stack.add_surface(ConicSurface("back", radius=-50.0, conicity=0.0, z0=5.0,
                               aperture=45.0), medium_index=1.5)
stack.add_media(1.0)

a = np.arange(128) - 64
env = np.exp(-a**2 / (2 * (128 * 0.1) ** 2))
pat = (env[None, :] * env[:, None]).astype(np.float32)
src = IrradianceSource(pat, pitch=1.0, origin=(0, 0, -20.0), direction=(0, 0, 1))

tr = stack.trace(src)
irr = irradiance_from_trace(tr, detector_z=38.0)
print(irr["irradiance"].max(), irr["valid"].mean())
```

`TraceResult` exposes `hits` (`S, N, 3`), `tir` (TIR flags), `blocked`
(aperture-masked rays), and `propagate_to(z)` for moving the final rays to any
detector plane.

## Examples

```bash
python examples/trace_lens.py            # scan for the focal plane, save PNGs
python examples/trace_lens.py --paraxial # use the thin-lens focal length
python examples/trace_lens.py --out ./figs
```

Writes `source_pattern.png` and `irradiance_focus.png` into the output directory.

## Tests

```bash
pytest -q          # physics + tracing + irradiance + offscreen GUI tests
python _verify.py  # end-to-end sanity checks (prints a summary)
```

The GUI tests run headless (offscreen Qt platform, core-profile surface).
