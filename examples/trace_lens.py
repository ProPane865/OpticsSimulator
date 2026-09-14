"""Trace a bi-convex lens and map the irradiance at (and around) focus.

Builds a simple two-surface bi-convex lens, launches a Gaussian-irradiance
source, and traces every ray through the stack with the Numba backend. It then
computes the output irradiance on the detector plane via the ray-map Jacobian
(``I_out = I_in / |det J|``) and saves the source pattern and a focal-plane
irradiance map to PNG.

Run::

    python examples/trace_lens.py
    python examples/trace_lens.py --paraxial     # use the thin-lens focal length
    python examples/trace_lens.py --out ./figs
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Allow running this script directly (``python examples/trace_lens.py``) from
# anywhere by putting the project root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics import (ConicSurface, OpticalStack, IrradianceSource,
                    irradiance_from_trace)


def build_lens(R=50.0, n=1.5, z1=5.0, aperture=70.0):
    """A bi-convex lens: convex front, concave back, both circularly clipped."""
    stack = OpticalStack("bi_convex")
    stack.add_surface(ConicSurface("front", radius=R, conicity=0.0, z0=0.0,
                                   aperture=aperture), medium_index=n)
    stack.add_surface(ConicSurface("back", radius=-R, conicity=0.0, z0=z1,
                                   aperture=aperture), medium_index=n)
    stack.add_media(1.0)
    return stack


def gaussian_source(grid=128, sigma_frac=0.25, peak=1.0, pitch=1.0, origin=(0, 0, -20.0)):
    a = np.arange(grid) - grid / 2.0
    env = peak * np.exp(-(a ** 2) / (2 * (grid * sigma_frac) ** 2))
    pat = (env[None, :] * env[:, None]).astype(np.float32)
    return IrradianceSource(pat, pitch=pitch, origin=origin, direction=(0, 0, 1))


def find_focal_plane(stack, src, z_range=(30, 90), n=121):
    """Return the detector z with the smallest RMS ray spot radius."""
    tr = stack.trace(src)
    zs = np.linspace(z_range[0], z_range[1], n)
    spots = []
    for z in zs:
        p, _ = tr.propagate_to(z)
        spots.append(np.sqrt(np.mean(p[:, 0] ** 2 + p[:, 1] ** 2)))
    spots = np.asarray(spots)
    return float(zs[np.argmin(spots)]), float(spots.min())


def save_png(arr, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(arr, origin="lower", cmap="inferno")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--out", type=str, default="examples/figs")
    ap.add_argument("--paraxial", action="store_true",
                    help="use the thin-lens paraxial focal length instead of scanning")
    args = ap.parse_args()

    stack = build_lens()
    src = gaussian_source(grid=args.grid, sigma_frac=0.1)
    print(f"source: {src.shape} grid, peak={src.pattern.max():.3f}, "
          f"total power={src.area:.3f} W")

    tr = stack.trace(src)
    print(f"trace: {tr.n_rays} rays through {tr.n_surfaces} surfaces "
          f"(blocked={int(tr.blocked.sum())})")

    if args.paraxial:
        fz = 50.0 / (1.5 - 1.0)  # thin-lens paraxial focal length R/(2(n-1))
        print(f"paraxial focal length: z={fz:.2f}")
    else:
        fz, spot = find_focal_plane(stack, src, z_range=(25, 60))
        print(f"focal plane: z={fz:.2f}  RMS spot={spot:.2f} px")

    irr = irradiance_from_trace(tr, detector_z=fz)
    peak = float(irr["irradiance"].max())
    valid = float(irr["valid"].mean())
    print(f"focus: output peak irradiance={peak:.2f} W, valid_frac={valid:.2f}")

    os.makedirs(args.out, exist_ok=True)
    save_png(src.pattern, os.path.join(args.out, "source_pattern.png"),
             "Source irradiance")
    save_png(irr["irradiance"], os.path.join(args.out, "irradiance_focus.png"),
             f"Irradiance at focus (z={fz:.1f})")


if __name__ == "__main__":
    main()
