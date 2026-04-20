"""Example 3 – 2-D FDTD point source in vacuum.

A Ricker-wavelet point source is placed at the centre of a 200×200 cell
free-space domain.  Circular wave fronts propagate outward and are
absorbed by the first-order Mur ABC at all four boundaries.

Run::

    python examples/example_2d_vacuum.py
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
dx = 1e-3          # 1 mm cells
f0 = 1.5e9         # 1.5 GHz dominant frequency

# ------------------------------------------------------------------ #
# Build simulation
# ------------------------------------------------------------------ #
sim = FDTD2D(Nx=Nx, Ny=Ny, dx=dx, boundary="mur")
src_x, src_y = Nx // 2, Ny // 2
sim.add_source(ix=src_x, iy=src_y, func=ricker_wavelet(f0=f0),
               source_type="soft")

# ------------------------------------------------------------------ #
# Run and save snapshots
# ------------------------------------------------------------------ #
snap_steps = [80, 160, 240, 320]
step = 0
snapshots = {}
for ts in snap_steps:
    sim.run(ts - step)
    step = ts
    snapshots[ts] = sim.Ez.copy()

# ------------------------------------------------------------------ #
# Plot 2×2 grid
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.ravel()

X, Y = np.meshgrid(sim.x * 1e3, sim.y * 1e3, indexing="ij")

for ax, ts in zip(axes, snap_steps):
    ez = snapshots[ts]
    vmax = np.abs(ez).max() or 1.0
    im = ax.pcolormesh(X, Y, ez, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       shading="auto")
    ax.set_title(f"Step {ts}  (t = {ts * sim.dt:.2e} s)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label=r"$E_z$ [V/m]")

fig.suptitle("2-D FDTD (TMz) – Ricker point source in vacuum", fontsize=13)
fig.tight_layout()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_2d_vacuum.png")
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.close(fig)
