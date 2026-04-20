"""Unit tests for the 2-D FDTD solver (fdtd.FDTD2D).

Tests cover:
    1. Construction defaults and shape checks.
    2. Field initialisation to zero.
    3. Mur ABC: energy must decrease after point source pulse exits.
    4. PEC boundary: Ez = 0 on all four walls.
    5. Material region: dielectric rectangle modifies the field pattern.
    6. Source injection (soft / hard).
    7. run_and_record: snapshot count and probe signal length.
    8. Symmetry: a point source at the centre should produce a
       field that is (approximately) symmetric in x and y.
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fdtd import FDTD2D
from fdtd.sources import ricker_wavelet, gaussian_pulse
from fdtd.constants import c0, eps0, mu0


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def total_energy_2d(sim: FDTD2D) -> float:
    """Approximate total EM energy in the 2-D domain."""
    u_E = 0.5 * eps0 * sim.eps_r * sim.Ez ** 2
    # Hx is shape (Nx+1, Ny); Hy is shape (Nx, Ny+1)
    # mu_r for Hx uses averaged y-adjacent eps nodes (same as in solver)
    mu_hx = 0.5 * (sim.mu_r[:, :-1] + sim.mu_r[:, 1:])
    mu_hy = 0.5 * (sim.mu_r[:-1, :] + sim.mu_r[1:, :])
    u_Hx = 0.5 * mu0 * mu_hx * sim.Hx ** 2
    u_Hy = 0.5 * mu0 * mu_hy * sim.Hy ** 2
    dV = sim.dx * sim.dy
    return float(np.sum(u_E) * dV + np.sum(u_Hx) * dV + np.sum(u_Hy) * dV)


# ------------------------------------------------------------------ #
# Construction
# ------------------------------------------------------------------ #

class TestConstruction:
    def test_default_shapes(self):
        sim = FDTD2D(Nx=50, Ny=60, dx=1e-3)
        assert sim.Ez.shape == (51, 61)
        assert sim.Hx.shape == (51, 60)
        assert sim.Hy.shape == (50, 61)

    def test_default_vacuum(self):
        sim = FDTD2D(Nx=30, Ny=30, dx=1e-3)
        np.testing.assert_allclose(sim.eps_r, np.ones((31, 31)))
        np.testing.assert_allclose(sim.mu_r, np.ones((31, 31)))

    def test_courant_dt(self):
        dx = 2e-3
        courant = 0.5 / np.sqrt(2)
        sim = FDTD2D(Nx=20, Ny=20, dx=dx, courant=courant)
        expected_dt = courant * dx / c0
        assert abs(sim.dt - expected_dt) < 1e-25

    def test_separate_dx_dy(self):
        sim = FDTD2D(Nx=20, Ny=30, dx=1e-3, dy=2e-3)
        assert sim.dx == 1e-3
        assert sim.dy == 2e-3

    def test_wrong_eps_shape_raises(self):
        with pytest.raises(ValueError):
            FDTD2D(Nx=20, Ny=20, dx=1e-3, eps_r=np.ones((5, 5)))

    def test_source_out_of_range_raises(self):
        sim = FDTD2D(Nx=20, Ny=20, dx=1e-3)
        with pytest.raises(ValueError):
            sim.add_source(ix=100, iy=5, func=lambda t: 0.0)


# ------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------ #

class TestInitialConditions:
    def test_fields_zero(self):
        sim = FDTD2D(Nx=30, Ny=30, dx=1e-3)
        np.testing.assert_array_equal(sim.Ez, 0.0)
        np.testing.assert_array_equal(sim.Hx, 0.0)
        np.testing.assert_array_equal(sim.Hy, 0.0)

    def test_time_zero(self):
        sim = FDTD2D(Nx=30, Ny=30, dx=1e-3)
        assert sim.n == 0
        assert sim.time == 0.0


# ------------------------------------------------------------------ #
# Mur ABC – energy absorption
# ------------------------------------------------------------------ #

class TestMurABC2D:
    def test_energy_decreases(self):
        Nx = Ny = 80
        dx = 1e-3
        f0 = 2e9

        sim = FDTD2D(Nx=Nx, Ny=Ny, dx=dx, boundary="mur")
        sim.add_source(ix=Nx // 2, iy=Ny // 2,
                       func=ricker_wavelet(f0=f0),
                       source_type="soft")

        # Run to fully inject the pulse
        tau = np.sqrt(6.0) / (np.pi * f0)
        n_inject = int(np.ceil(3 * tau / sim.dt)) + 20
        sim.run(n_inject)
        e0 = total_energy_2d(sim)

        # Run much longer so the wave hits all four walls and exits
        sim.run(Nx * 5)
        e1 = total_energy_2d(sim)

        assert e1 < e0 * 0.5, (
            f"Energy should drop after pulse exits Mur domain; got {e1:.3e} vs {e0:.3e}"
        )


# ------------------------------------------------------------------ #
# PEC boundaries
# ------------------------------------------------------------------ #

class TestPECBoundary2D:
    def test_boundary_ez_zero(self):
        Nx = Ny = 40
        sim = FDTD2D(Nx=Nx, Ny=Ny, dx=1e-3, boundary="pec")
        f0 = 2e9
        sim.add_source(ix=Nx // 2, iy=Ny // 2,
                       func=ricker_wavelet(f0=f0))
        sim.run(50)
        np.testing.assert_array_equal(sim.Ez[0, :], 0.0)
        np.testing.assert_array_equal(sim.Ez[-1, :], 0.0)
        np.testing.assert_array_equal(sim.Ez[:, 0], 0.0)
        np.testing.assert_array_equal(sim.Ez[:, -1], 0.0)


# ------------------------------------------------------------------ #
# Source injection
# ------------------------------------------------------------------ #

class TestSourceInjection2D:
    def test_soft_source_activates_field(self):
        sim = FDTD2D(Nx=40, Ny=40, dx=1e-3, boundary="mur")
        sim.add_source(ix=20, iy=20, func=lambda t: 1.0, source_type="soft")
        sim.step()
        assert sim.Ez[20, 20] != 0.0

    def test_hard_source_sets_field(self):
        sim = FDTD2D(Nx=40, Ny=40, dx=1e-3, boundary="mur")
        val = 0.42
        sim.add_source(ix=20, iy=20, func=lambda t: val, source_type="hard")
        sim.step()
        assert abs(sim.Ez[20, 20] - val) < 1e-12


# ------------------------------------------------------------------ #
# Symmetry (point source at centre)
# ------------------------------------------------------------------ #

class TestSymmetry2D:
    def test_fourfold_symmetry(self):
        """Ez should be symmetric under 90° rotation when source is centred."""
        N = 60
        dx = 1e-3
        f0 = 2e9

        sim = FDTD2D(Nx=N, Ny=N, dx=dx, boundary="pec")
        sim.add_source(ix=N // 2, iy=N // 2,
                       func=ricker_wavelet(f0=f0), source_type="soft")
        sim.run(30)

        ez = sim.Ez
        # 90° rotation: Ez[i,j] ≈ Ez[j, N-i] for a symmetric domain
        # Use the transpose as a proxy (symmetry under i↔j)
        diff = np.abs(ez - ez.T)
        rel_diff = diff.max() / (np.abs(ez).max() + 1e-30)
        assert rel_diff < 0.02, (
            f"Ez should be approximately symmetric; max relative diff = {rel_diff:.3e}"
        )


# ------------------------------------------------------------------ #
# Dielectric region
# ------------------------------------------------------------------ #

class TestDielectric2D:
    def test_dielectric_modifies_field(self):
        """Ez inside and outside the dielectric should differ significantly."""
        N = 80
        dx = 1e-3
        f0 = 2e9
        eps_r_val = 4.0

        # Dielectric in the upper-right quadrant
        eps_r = np.ones((N + 1, N + 1))
        eps_r[N // 2:, N // 2:] = eps_r_val

        sim_vac = FDTD2D(Nx=N, Ny=N, dx=dx, boundary="pec")
        sim_vac.add_source(ix=N // 4, iy=N // 4,
                           func=ricker_wavelet(f0=f0), source_type="soft")
        sim_vac.run(60)

        sim_eps = FDTD2D(Nx=N, Ny=N, dx=dx, eps_r=eps_r, boundary="pec")
        sim_eps.add_source(ix=N // 4, iy=N // 4,
                           func=ricker_wavelet(f0=f0), source_type="soft")
        sim_eps.run(60)

        # Fields in the dielectric region should differ from vacuum
        region = (slice(N // 2 + 5, N - 5), slice(N // 2 + 5, N - 5))
        diff = np.abs(sim_eps.Ez[region] - sim_vac.Ez[region])
        assert diff.max() > 1e-6, (
            "Dielectric should alter the field inside the region"
        )


# ------------------------------------------------------------------ #
# run_and_record
# ------------------------------------------------------------------ #

class TestRunAndRecord2D:
    def test_snapshot_count(self):
        sim = FDTD2D(Nx=30, Ny=30, dx=1e-3)
        snaps, _ = sim.run_and_record(50, record_interval=10)
        assert len(snaps) == 5

    def test_probe_length(self):
        sim = FDTD2D(Nx=30, Ny=30, dx=1e-3)
        _, ts = sim.run_and_record(40, record_interval=4,
                                   probe_positions=[(15, 15)])
        assert len(ts[(15, 15)]) == 10

    def test_snapshot_shape(self):
        Nx, Ny = 30, 40
        sim = FDTD2D(Nx=Nx, Ny=Ny, dx=1e-3)
        snaps, _ = sim.run_and_record(10)
        assert snaps[0].shape == (Nx + 1, Ny + 1)


# ------------------------------------------------------------------ #
# Coordinate helpers
# ------------------------------------------------------------------ #

class TestCoordinates2D:
    def test_x_length(self):
        sim = FDTD2D(Nx=30, Ny=40, dx=2e-3)
        assert len(sim.x) == 31

    def test_y_length(self):
        sim = FDTD2D(Nx=30, Ny=40, dx=2e-3, dy=3e-3)
        assert len(sim.y) == 41

    def test_x_spacing(self):
        dx = 2e-3
        sim = FDTD2D(Nx=30, Ny=20, dx=dx)
        np.testing.assert_allclose(np.diff(sim.x), dx)
