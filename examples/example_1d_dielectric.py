"""Example 2 – 1-D FDTD with a dielectric slab.

A Gaussian pulse is launched from the left side of a 400-cell domain.
The right half of the domain is filled with a dielectric material
(εᵣ = 4, i.e. n = 2).  The simulation demonstrates partial reflection
at the dielectric interface and the reduced phase velocity inside the slab.

Run::

    python examples/example_1d_dielectric.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fdtd import FDTD1D
from fdtd.sources import gaussian_pulse
from fdtd.constants import c0

# ------------------------------------------------------------------ #
# Simulation parameters
# ------------------------------------------------------------------ #
N = 400
dx = 1e-3               # 1 mm
eps_r_slab = 4.0        # relative permittivity of slab (n = 2)
interface = N // 2      # interface at cell N/2

# Build permittivity profile: vacuum (1.0) left, slab (4.0) right
eps_r = np.ones(N + 1)
eps_r[interface:] = eps_r_slab

# Source parameters: Gaussian pulse well before the interface
tau = 30 * dx / c0      # pulse width ~ 30 cells of travel time
t0 = 3 * tau            # delay so pulse starts from near-zero amplitude
src_pos = N // 6        # source node index

# ------------------------------------------------------------------ #
# Build simulation
# ------------------------------------------------------------------ #
sim = FDTD1D(N=N, dx=dx, eps_r=eps_r, boundary="mur")
sim.add_source(pos=src_pos, func=gaussian_pulse(t0=t0, tau=tau),
               source_type="soft")

# ------------------------------------------------------------------ #
# Run and collect snapshots
# ------------------------------------------------------------------ #
snap_steps = [200, 400, 550, 700]
step = 0
snapshots = {}
for ts in snap_steps:
    sim.run(ts - step)
    step = ts
    snapshots[ts] = sim.Ez.copy()

# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(len(snap_steps), 1, figsize=(10, 10), sharex=True)
x_mm = sim.x_Ez * 1e3
x_int = interface * dx * 1e3

for ax, ts in zip(axes, snap_steps):
    ax.plot(x_mm, snapshots[ts], "b-", lw=1.5, label=r"$E_z$ [V/m]")
    ax.axvline(x_int, color="k", ls="--", lw=1, label=f"interface (εᵣ={eps_r_slab})")
    ax.axvspan(x_int, x_mm[-1], alpha=0.08, color="orange", label="dielectric")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Step {ts}  (t = {ts * sim.dt:.2e} s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, ls=":")

axes[-1].set_xlabel("x [mm]")
fig.suptitle(
    f"1-D FDTD – Gaussian pulse + dielectric slab (εᵣ = {eps_r_slab})", fontsize=13
)
fig.tight_layout()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_1d_dielectric.png")
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.close(fig)
