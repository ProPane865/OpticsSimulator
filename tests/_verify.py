import numpy as np
from optics import (accel, ConicSurface, Plane, PolySagSurface, OpticalStack,
                    IrradianceSource, refract, irradiance_from_trace)

print("HAVE_NUMBA:", accel.HAVE_NUMBA)

# 1) Snell: n1*sin(theta1) = n2*sin(theta2)
d = np.array([[0.3, 0.0, np.sqrt(1 - 0.09)]])
n = np.array([[0.0, 0.0, 1.0]])
dt, ci, tir = refract(d, n, 1.0, 1.5)
sin1 = 0.3
sin2 = np.sqrt(dt[0, 0] ** 2 + dt[0, 1] ** 2)
print(f"[snell] n1*sin1={1.0*sin1:.5f}  n2*sin2={1.5*sin2:.5f}  tir={tir[0]}")
assert abs(1.0 * sin1 - 1.5 * sin2) < 1e-9

# TIR check: dense->rare
d2 = np.array([[np.sin(np.radians(60)), 0.0, -np.cos(np.radians(60))]])
n2 = np.array([[0.0, 0.0, 1.0]])
_, _, tir2 = refract(d2, n2, 1.5, 1.0)
nc = np.sin(np.radians(90) ) * 1.5  # critical sin
print(f"[tir] critical_sin={1.5*1.0:.5f}, tir at 60deg={tir2[0]} (expect True)")
assert tir2[0]

# 2) Conic surface normal vs finite-difference (poly check)
s = ConicSurface("lens", radius=100.0, conicity=-0.5, z0=0.0)
x = np.array([[10.0, 5.0]]); y = np.array([[5.0, 8.0]])
gx, gy = s.gradient(x, y)
h = 1e-4
sag = lambda a, b: s.sag(a, b)
gxn = (sag(x + h, y) - sag(x - h, y)) / (2 * h)
gyn = (sag(x, y + h) - sag(x, y - h)) / (2 * h)
print(f"[grad] analytic={gx[0,0]:.6f},{gy[0,0]:.6f}  fd={gxn[0,0]:.6f},{gyn[0,0]:.6f}")
assert np.allclose(gx, gxn, atol=1e-4) and np.allclose(gy, gyn, atol=1e-4)

# 3) PolySag matches a conic-ish surface & normal finite-diff
P = PolySagSurface("poly", order=4)
P.coeffs[2, 0] = s.curvature / 2.0   # (c/2) x^2 term ~ sag c r^2/2 (paraxial)
P.coeffs[0, 2] = s.curvature / 2.0
gx2, gy2 = P.gradient(x, y)
gxn2 = (P.sag(x + h, y) - P.sag(x - h, y)) / (2 * h)
print(f"[poly] analytic gx={gx2[0,0]:.6f}  fd={gxn2[0,0]:.6f}")
assert np.allclose(gx2, gxn2, atol=1e-3)

# 4) End-to-end trace + irradiance mapping (simple bi-convex lens)
R = 50.0
lens = ConicSurface("lens", radius=R, conicity=0.0, z0=0.0)
stack = OpticalStack("lens")
stack.add_surface(lens, medium_index=1.5)
stack.add_surface(ConicSurface("lens2", radius=-R, conicity=0.0, z0=5.0), medium_index=1.5)
stack.add_media(1.0)
grid = 80
pat = np.exp(-((np.arange(grid) - grid / 2) ** 2) / (2 * (grid / 5) ** 2))
pat = pat[None, :] * pat[:, None]  # Gaussian blob
src = IrradianceSource(pat, pitch=1.0, origin=(0, 0, -20.0), direction=(0, 0, 1))
tr = stack.trace(src)
f = R / (2 * (1.5 - 1))  # rough paraxial focus
print(f"[trace] n_rays={tr.n_rays} n_surfaces={tr.n_surfaces} numba={accel.HAVE_NUMBA}")
irr = irradiance_from_trace(tr, detector_z=f)
print(f"[irradiance] peak={irr['irradiance'].max():.2f}  valid_frac={irr['valid'].mean():.2f}")
assert irr["irradiance"].max() > 0 and irr["valid"].mean() > 0.5

# 5) Numba backend focuses a collimated beam: find focal plane, verify compression
assert accel.HAVE_NUMBA is True
print(f"[numba] backend active = {accel.HAVE_NUMBA}")

in_peak = pat.max()
pts0 = tr.hits[-1]  # ray exit points on last surface (N,3)
zs = np.linspace(10, 90, 161)
spot = []
for z in zs:
    p, _ = tr.propagate_to(z)
    r2 = np.mean(p[:, 0] ** 2 + p[:, 1] ** 2)
    spot.append(np.sqrt(r2))
spot = np.array(spot)
fz = float(zs[np.argmin(spot)])
print(f"[focus] focal z={fz:.2f}  min_spot_rms={spot.min():.3f} px")
irr = irradiance_from_trace(tr, detector_z=fz)
out_peak = irr["irradiance"].max()
# Physical focusing: ray spot at focus is smaller than at the lens exit plane.
spot_focus = float(spot.min())
spot_lens = float(np.sqrt(np.mean(pts0[:, 0] ** 2 + pts0[:, 1] ** 2)))
print(f"[focus] spot_lens={spot_lens:.2f} px  spot_focus={spot_focus:.2f} px  peak_gain={out_peak/in_peak:.1f}x")
assert spot_focus < spot_lens and out_peak > in_peak

# 6) Aperture clipping: rays beyond the lens radius are blocked and excluded
stack2 = OpticalStack("ap")
stack2.add_surface(ConicSurface("l1", radius=50.0, conicity=0.0, z0=0.0, aperture=40.0),
                   medium_index=1.5)
stack2.add_surface(ConicSurface("l2", radius=-50.0, conicity=0.0, z0=5.0, aperture=40.0),
                   medium_index=1.5)
stack2.add_media(1.0)
src2 = IrradianceSource(pat, pitch=1.0, origin=(0, 0, -20.0))
tr2 = stack2.trace(src2)
n_blocked = int(tr2.blocked.sum())
print(f"[aperture] blocked_rays={n_blocked}/{tr2.n_rays}")
assert n_blocked > 0

print("\nALL CHECKS PASSED")
