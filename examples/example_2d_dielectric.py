"""Example 4 – 2-D FDTD scattering off a dielectric rectangle.

A Ricker-wavelet point source placed in the lower-left quadrant excites a
wave that propagates toward a dielectric rectangle (εᵣ = 4) placed in the
centre of the domain.  Refraction and partial reflection at the rectangle
boundaries are visible in the Ez snapshot.

Run::

    python examples/example_2d_dielectric.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fdtd import FDTD2D
from fdtd.sources import ricker_wavelet

# ------------------------------------------------------------------ #
# Simulation parameters
# ------------------------------------------------------------------ #
Nx, Ny = 200, 200
dx = 1e-3
f0 = 1.5e9

# Dielectric rectangle: cells [80:120, 80:120]
eps_r_val = 4.0
ix_lo, ix_hi = 80, 120   # x-node indices
iy_lo, iy_hi = 80, 120   # y-node indices

eps_r = np.ones((Nx + 1, Ny + 1))
eps_r[ix_lo:ix_hi + 1, iy_lo:iy_hi + 1] = eps_r_val

# Source in the lower-left quadrant
src_x, src_y = Nx // 4, Ny // 4

# ------------------------------------------------------------------ #
# Build simulation
# ------------------------------------------------------------------ #
sim = FDTD2D(Nx=Nx, Ny=Ny, dx=dx, eps_r=eps_r, boundary="mur")
sim.add_source(ix=src_x, iy=src_y, func=ricker_wavelet(f0=f0),
               source_type="soft")

# ------------------------------------------------------------------ #
# Run and collect snapshots
# ------------------------------------------------------------------ #
snap_steps = [100, 200, 300, 400]
step = 0
snapshots = {}
for ts in snap_steps:
    sim.run(ts - step)
    step = ts
    snapshots[ts] = sim.Ez.copy()

# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.ravel()

X, Y = np.meshgrid(sim.x * 1e3, sim.y * 1e3, indexing="ij")

# Build dielectric mask for overlay
mask = eps_r > 1.0          # True inside slab

for ax, ts in zip(axes, snap_steps):
    ez = snapshots[ts]
    vmax = np.abs(ez).max() or 1.0
    im = ax.pcolormesh(X, Y, ez, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       shading="auto")
    # Overlay dielectric region
    ax.contour(X, Y, mask.astype(float), levels=[0.5], colors="k",
               linewidths=1.5)
    ax.set_title(f"Step {ts}  (t = {ts * sim.dt:.2e} s)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label=r"$E_z$ [V/m]")

fig.suptitle(
    f"2-D FDTD (TMz) – Ricker source + dielectric obstacle (εᵣ = {eps_r_val})",
    fontsize=13,
)
fig.tight_layout()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_2d_dielectric.png")
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.close(fig)
