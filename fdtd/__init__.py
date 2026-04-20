"""FDTD – Finite-Difference Time-Domain simulation package.

Implements the Yee staggered-grid algorithm for characterising
electric and magnetic fields in 1-D and 2-D.

Quick start::

    from fdtd import FDTD1D, FDTD2D
    from fdtd.sources import ricker_wavelet, gaussian_pulse

    # 1-D free-space propagation
    sim = FDTD1D(N=300, dx=1e-3)
    sim.add_source(pos=100, func=ricker_wavelet(f0=1e9))
    sim.run(500)
    sim.plot_fields()

    # 2-D point source
    sim2 = FDTD2D(Nx=200, Ny=200, dx=1e-3)
    sim2.add_source(ix=100, iy=100, func=ricker_wavelet(f0=1e9))
    sim2.run(300)
    sim2.plot_Ez()
"""

from .fdtd1d import FDTD1D
from .fdtd2d import FDTD2D
from . import sources
from . import constants

__all__ = ["FDTD1D", "FDTD2D", "sources", "constants"]
