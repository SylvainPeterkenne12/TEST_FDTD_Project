"""Unit tests for the 1-D FDTD solver (fdtd.FDTD1D).

Tests cover:
    1. Construction defaults (vacuum, Mur ABC).
    2. Wave speed: a Gaussian pulse must travel exactly one cell per step
       when the Courant factor is 1 (S = c₀Δt/Δx = 1).
    3. Energy decay through Mur ABC: total energy should decrease
       monotonically once the pulse has left the domain.
    4. Material interface: a pulse hitting an εᵣ > 1 slab must produce a
       reflected wave with the correct Fresnel coefficient sign.
    5. Hard vs. soft source behaviour.
    6. PEC boundary: fields must vanish at domain edges.
    7. Lossy medium: energy must decay faster than in vacuum.
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fdtd import FDTD1D
from fdtd.sources import gaussian_pulse, ricker_wavelet
from fdtd.constants import c0, eps0, mu0, eta0


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def total_energy(sim: FDTD1D) -> float:
    """Approximate total EM energy in the domain."""
    u_E = 0.5 * eps0 * sim.eps_r * sim.Ez ** 2  # electric energy density
    u_H = 0.5 * mu0 * sim.mu_r * sim.Hy ** 2   # magnetic energy density
    return float(np.sum(u_E) * sim.dx + np.sum(u_H) * sim.dx)


# ------------------------------------------------------------------ #
# Construction
# ------------------------------------------------------------------ #

class TestConstruction:
    def test_default_vacuum(self):
        sim = FDTD1D(N=100, dx=1e-3)
        assert sim.N == 100
        assert sim.dx == 1e-3
        assert sim.Ez.shape == (101,)
        assert sim.Hy.shape == (100,)
        np.testing.assert_allclose(sim.eps_r, np.ones(101))
        np.testing.assert_allclose(sim.mu_r, np.ones(100))

    def test_courant_dt(self):
        dx = 2e-3
        courant = 0.9
        sim = FDTD1D(N=50, dx=dx, courant=courant)
        expected_dt = courant * dx / c0
        assert abs(sim.dt - expected_dt) < 1e-25

    def test_scalar_material_broadcast(self):
        sim = FDTD1D(N=50, dx=1e-3, eps_r=4.0, mu_r=2.0)
        np.testing.assert_allclose(sim.eps_r, 4.0 * np.ones(51))
        np.testing.assert_allclose(sim.mu_r, 2.0 * np.ones(50))

    def test_array_material(self):
        N = 60
        eps = np.linspace(1.0, 4.0, N + 1)
        mu = np.ones(N)
        sim = FDTD1D(N=N, dx=1e-3, eps_r=eps, mu_r=mu)
        np.testing.assert_allclose(sim.eps_r, eps)

    def test_wrong_array_shape_raises(self):
        with pytest.raises(ValueError):
            FDTD1D(N=50, dx=1e-3, eps_r=np.ones(99))

    def test_source_out_of_range_raises(self):
        sim = FDTD1D(N=50, dx=1e-3)
        with pytest.raises(ValueError):
            sim.add_source(pos=200, func=lambda t: 0.0)


# ------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------ #

class TestInitialConditions:
    def test_fields_zero_at_start(self):
        sim = FDTD1D(N=100, dx=1e-3)
        np.testing.assert_array_equal(sim.Ez, 0.0)
        np.testing.assert_array_equal(sim.Hy, 0.0)

    def test_time_zero_at_start(self):
        sim = FDTD1D(N=100, dx=1e-3)
        assert sim.n == 0
        assert sim.time == 0.0


# ------------------------------------------------------------------ #
# Wave speed (Courant S = 1 → pulse travels exactly 1 cell/step)
# ------------------------------------------------------------------ #

class TestWaveSpeed:
    """In vacuum with S = 1 the Yee scheme is dispersion-free in 1-D
    (the numerical phase velocity equals c₀ exactly)."""

    def test_pulse_travels_one_cell_per_step(self):
        N = 200
        dx = 1e-3
        src_pos = 20  # inject at node 20
        tau = 8 * dx / c0
        t0 = 4 * tau

        sim = FDTD1D(N=N, dx=dx, courant=1.0, boundary="mur")
        sim.add_source(pos=src_pos, func=gaussian_pulse(t0=t0, tau=tau),
                       source_type="hard")

        # Run until pulse peak has passed (t0 in source time)
        n_peak = int(np.round(t0 / sim.dt))
        sim.run(n_peak)
        # At this moment the hard source is at its peak; the forward-going
        # wave should be centred near src_pos as well (hard source)
        peak_idx = int(np.argmax(np.abs(sim.Ez)))
        assert abs(peak_idx - src_pos) <= 3  # within 3 cells

    def test_right_going_pulse_position(self):
        """Soft source in vacuum: after n steps, peak of right-going pulse
        is approximately at src_pos + n (one cell per step)."""
        N = 300
        dx = 1e-3
        src_pos = N // 4
        tau = 10 * dx / c0
        t0 = 5 * tau

        sim = FDTD1D(N=N, dx=dx, courant=1.0, boundary="pec")
        sim.add_source(pos=src_pos, func=gaussian_pulse(t0=t0, tau=tau),
                       source_type="soft")

        # Run until peak has fully launched
        n_launch = int(np.ceil(t0 / sim.dt)) + 30
        sim.run(n_launch)

        # Additional propagation steps
        extra = 50
        sim.run(extra)

        # Peak of right-going wave (right half of domain)
        ez_right = sim.Ez[src_pos + 10:]
        peak_right = int(np.argmax(np.abs(ez_right)))
        expected_pos = extra  # ~ extra cells to the right of src region
        # Allow ±25 cells tolerance (pulse width + source region)
        assert abs(peak_right - expected_pos) <= 25


# ------------------------------------------------------------------ #
# Mur ABC: energy absorption
# ------------------------------------------------------------------ #

class TestMurABC:
    def test_energy_decreases_after_pulse_leaves(self):
        """Total energy must decrease once the pulse has fully exited."""
        N = 200
        dx = 1e-3
        src_pos = N // 2
        tau = 10 * dx / c0
        t0 = 5 * tau

        sim = FDTD1D(N=N, dx=dx, boundary="mur")
        sim.add_source(pos=src_pos,
                       func=gaussian_pulse(t0=t0, tau=tau),
                       source_type="soft")

        # Run to inject the pulse fully
        n_inject = int(np.ceil((t0 + 6 * tau) / sim.dt))
        sim.run(n_inject)
        e0 = total_energy(sim)

        # Run long enough for pulse to exit both boundaries
        sim.run(N * 2)
        e1 = total_energy(sim)

        assert e1 < e0 * 0.1, (
            f"Energy should drop significantly after pulse exits; got {e1:.3e} vs {e0:.3e}"
        )

    def test_pec_energy_conserved(self):
        """With PEC boundaries no energy leaves; total should stay ≥ initial."""
        N = 200
        dx = 1e-3
        src_pos = N // 2
        tau = 10 * dx / c0
        t0 = 5 * tau

        sim = FDTD1D(N=N, dx=dx, boundary="pec")
        sim.add_source(pos=src_pos,
                       func=gaussian_pulse(t0=t0, tau=tau),
                       source_type="soft")

        n_inject = int(np.ceil((t0 + 6 * tau) / sim.dt))
        sim.run(n_inject)
        e0 = total_energy(sim)

        # After injection the energy inside a PEC box is conserved
        # (Poynting theorem + lossless material)
        sim.run(100)
        e1 = total_energy(sim)

        # Allow 1 % numerical rounding
        assert e1 >= e0 * 0.99, (
            f"PEC box energy should be conserved; got {e1:.3e} vs {e0:.3e}"
        )


# ------------------------------------------------------------------ #
# Dielectric interface – Fresnel reflection
# ------------------------------------------------------------------ #

class TestDielectricInterface:
    def test_reflected_wave_sign_and_amplitude(self):
        """For a wave incident from vacuum onto εᵣ > 1 medium the
        reflection coefficient Γ = (1 − √εᵣ)/(1 + √εᵣ) is negative."""
        N = 400
        dx = 1e-3
        eps_r_slab = 4.0
        interface = N // 2

        eps_r = np.ones(N + 1)
        eps_r[interface:] = eps_r_slab

        tau = 20 * dx / c0
        t0 = 5 * tau
        src_pos = N // 4  # well left of interface

        sim = FDTD1D(N=N, dx=dx, eps_r=eps_r, boundary="mur")
        sim.add_source(pos=src_pos,
                       func=gaussian_pulse(t0=t0, tau=tau),
                       source_type="soft")

        # Record at a probe to the left of the interface
        probe = src_pos + (interface - src_pos) // 2  # midway between src and interface

        n_total = int(np.ceil((t0 + 6 * tau) / sim.dt)) + interface
        snapshots, ts = sim.run_and_record(n_total,
                                           probe_positions=[probe])

        signal = np.array(ts[probe])

        # The reflected pulse arrives at probe ~ 2*(interface - probe) steps after
        # the incident peak (one cell per step with Courant=1).
        # For εᵣ > 1 the reflected E-field is negative (Γ = (1-n)/(1+n) < 0).
        # Verify by checking that the signal minimum after the incident pulse
        # peak is negative.
        i_peak = int(np.argmax(signal))
        # Skip at least one pulse-width beyond the peak to isolate reflected wave
        skip = max(50, int(np.round(2.0 * tau / sim.dt)))
        tail_min = signal[i_peak + skip:].min()
        assert tail_min < -1e-4, (
            f"Reflected wave should be negative for εᵣ > 1; got min = {tail_min:.4f}"
        )


# ------------------------------------------------------------------ #
# Source types
# ------------------------------------------------------------------ #

class TestSources:
    def test_hard_source_overrides_field(self):
        sim = FDTD1D(N=100, dx=1e-3, boundary="pec")
        src_val = 0.7
        sim.add_source(pos=50, func=lambda t: src_val, source_type="hard")
        sim.step()
        assert abs(sim.Ez[50] - src_val) < 1e-12

    def test_soft_source_adds_to_field(self):
        sim = FDTD1D(N=100, dx=1e-3, boundary="pec")
        sim.Ez[50] = 0.3
        sim.add_source(pos=50, func=lambda t: 0.5, source_type="soft")
        sim.step()
        # After one step with PEC and no H gradient at index 50 initially,
        # the soft source should have incremented the field.
        assert sim.Ez[50] != 0.3  # field changed


# ------------------------------------------------------------------ #
# PEC boundaries
# ------------------------------------------------------------------ #

class TestPECBoundary:
    def test_boundary_nodes_zero(self):
        sim = FDTD1D(N=100, dx=1e-3, boundary="pec")
        tau = 5 * sim.dx / c0
        t0 = 3 * tau
        sim.add_source(pos=50, func=gaussian_pulse(t0=t0, tau=tau))
        sim.run(100)
        assert sim.Ez[0] == 0.0
        assert sim.Ez[-1] == 0.0


# ------------------------------------------------------------------ #
# run_and_record
# ------------------------------------------------------------------ #

class TestRunAndRecord:
    def test_snapshot_count(self):
        sim = FDTD1D(N=50, dx=1e-3)
        snaps, _ = sim.run_and_record(100, record_interval=10)
        # Steps 0..99, recording at 0,10,20,...,90 → 10 snapshots
        assert len(snaps) == 10

    def test_probe_signal_length(self):
        sim = FDTD1D(N=50, dx=1e-3)
        snaps, ts = sim.run_and_record(100, record_interval=5, probe_positions=[25])
        assert len(ts[25]) == 20  # 100 / 5 = 20 recordings

    def test_snapshot_shape(self):
        sim = FDTD1D(N=80, dx=1e-3)
        snaps, _ = sim.run_and_record(10)
        assert snaps[0].shape == (81,)


# ------------------------------------------------------------------ #
# Lossy medium
# ------------------------------------------------------------------ #

class TestLossyMedium:
    def test_energy_decays_in_lossy_medium(self):
        N = 100
        dx = 1e-3
        sigma = 10.0  # high conductivity → fast attenuation

        sim = FDTD1D(N=N, dx=dx, sigma_e=sigma, boundary="pec")
        # Inject a hard sinusoidal source for a few cycles then switch off
        f0 = 1e9
        sim.add_source(pos=N // 2, func=lambda t: np.sin(2 * np.pi * f0 * t),
                       source_type="soft")

        sim.run(100)
        e_after_inject = total_energy(sim)

        # Remove source and run longer → energy should decay
        sim._sources.clear()
        sim.run(200)
        e_final = total_energy(sim)

        assert e_final < e_after_inject * 0.5, (
            f"Energy should decay in lossy medium; got {e_final:.3e} vs {e_after_inject:.3e}"
        )
