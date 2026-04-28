"""Example 1 – 1-D FDTD in vacuum.

A Ricker-wavelet point source is placed near the left quarter of a
300-cell free-space grid.  The simulation is run for 500 time steps and
the resulting field at several instants is saved to a PNG figure.

Run::

    python examples/example_1d_vacuum.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fdtd import FDTD1D
from fdtd.sources import ricker_wavelet

# ------------------------------------------------------------------ #
# Simulation parameters
# ------------------------------------------------------------------ #
N = 300          # number of cells
dx = 1e-3        # cell size [m] → 1 mm
f0 = 1.5e9       # dominant frequency [Hz] → 1.5 GHz
src_pos = N // 4 # source position

# ------------------------------------------------------------------ #
# Build simulation
# ------------------------------------------------------------------ #
sim = FDTD1D(N=N, dx=dx, boundary="mur")
sim.add_source(pos=src_pos, func=ricker_wavelet(f0=f0), source_type="soft")

# ------------------------------------------------------------------ #
# Run and collect snapshots
# ------------------------------------------------------------------ #
n_total = 600
snap_steps = [100, 200, 300, 400, 500, 600]
snapshots = {}
step = 0
for target in snap_steps:
    sim.run(target - step)
    step = target
    snapshots[target] = (sim.Ez.copy(), sim.Hy.copy())

# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(len(snap_steps), 1, figsize=(10, 12), sharex=True)
x = sim.x_Ez

for ax, ts in zip(axes, snap_steps):
    ez, hy = snapshots[ts]
    ax.plot(x * 1e3, ez, "b-", lw=1.2, label=r"$E_z$ [V/m]")
    hy_x = sim.x_Hy * 1e3
    ax.plot(hy_x, hy * 376.73, "r--", lw=1.2,
            label=r"$\eta_0 H_y$ [V/m]")
    ax.axvline(src_pos * dx * 1e3, color="gray", ls=":", lw=0.8, label="source")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Step {ts}  (t = {ts * sim.dt:.2e} s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, ls=":")

axes[-1].set_xlabel("x [mm]")
fig.suptitle("1-D FDTD – Ricker pulse in vacuum (Mur ABC)", fontsize=13)
fig.tight_layout()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_1d_vacuum.png")
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.close(fig)
