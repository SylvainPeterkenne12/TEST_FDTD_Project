"""1D FDTD simulation using the Yee algorithm.

The 1D model simulates electromagnetic plane-wave propagation along the
x-axis.  Using the TMz convention the two coupled field components are:

* **Ez** – electric field polarised in z  (stored at *integer* x-nodes)
* **Hy** – magnetic field polarised in y  (stored at *half-integer* x-nodes)

Grid layout (N cells, N+1 Ez nodes, N Hy nodes)::

    Ez[0]  Hy[0]  Ez[1]  Hy[1]  ...  Ez[N-1]  Hy[N-1]  Ez[N]
    |---dx---|---dx---|              |---dx---|---dx---|

Yee update cycle per time step Δt:
    1. H update (half-step forward):
       Hy[k] += (Δt / (μ₀ μᵣ Δx)) * (Ez[k+1] - Ez[k])
    2. E update (full step):
       Ez[k] += (Δt / (ε₀ εᵣ Δx)) * (Hy[k] - Hy[k-1])   for interior k
       Boundary nodes treated with selected BC (Mur ABC or PEC).
"""

import numpy as np
import matplotlib.pyplot as plt

from .constants import c0, eps0, mu0


class FDTD1D:
    """1D FDTD simulation using the Yee staggered-grid algorithm.

    Parameters
    ----------
    N : int
        Number of cells (Ez has N+1 nodes, Hy has N nodes).
    dx : float
        Cell size [m].
    eps_r : float or array_like, shape (N+1,), optional
        Relative permittivity at each Ez node (default: 1.0 = vacuum).
    mu_r : float or array_like, shape (N,), optional
        Relative permeability at each Hy node (default: 1.0 = vacuum).
    sigma_e : float or array_like, shape (N+1,), optional
        Electric conductivity [S/m] at each Ez node (default: 0.0).
    courant : float, optional
        Courant stability factor S = c₀ Δt / Δx (default 1.0 – the
        theoretical maximum for 1-D).
    boundary : {'mur', 'pec'}
        Absorbing boundary ('mur', default) or perfect electric conductor.

    Attributes
    ----------
    Ez : ndarray, shape (N+1,)
        Electric field [V/m].
    Hy : ndarray, shape (N,)
        Magnetic field [A/m].
    n : int
        Current time-step index.
    dt : float
        Time step [s].
    """

    def __init__(
        self,
        N: int,
        dx: float,
        eps_r=1.0,
        mu_r=1.0,
        sigma_e=0.0,
        courant: float = 1.0,
        boundary: str = "mur",
    ):
        self.N = N
        self.dx = dx
        self.courant = courant
        self.boundary = boundary

        # Time step satisfying the Courant condition
        self.dt = courant * dx / c0

        # ------------------------------------------------------------------ #
        # Material arrays
        # ------------------------------------------------------------------ #
        self.eps_r = self._broadcast(eps_r, N + 1)   # at Ez nodes
        self.mu_r = self._broadcast(mu_r, N)          # at Hy nodes
        self.sigma_e = self._broadcast(sigma_e, N + 1)  # electric conductivity

        # ------------------------------------------------------------------ #
        # Precompute update coefficients
        # ------------------------------------------------------------------ #
        self._compute_coefficients()

        # ------------------------------------------------------------------ #
        # Field arrays
        # ------------------------------------------------------------------ #
        self.Ez = np.zeros(N + 1, dtype=float)
        self.Hy = np.zeros(N, dtype=float)

        # ------------------------------------------------------------------ #
        # Mur ABC storage
        # ------------------------------------------------------------------ #
        self._ez_left_old = 0.0   # Ez[1] at previous time step
        self._ez_right_old = 0.0  # Ez[N-1] at previous time step

        # ------------------------------------------------------------------ #
        # Source list
        # ------------------------------------------------------------------ #
        self._sources: list[dict] = []

        # Time step counter
        self.n = 0

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _broadcast(val, size: int) -> np.ndarray:
        if np.isscalar(val):
            return float(val) * np.ones(size)
        arr = np.asarray(val, dtype=float)
        if arr.shape != (size,):
            raise ValueError(
                f"Expected scalar or array of length {size}, got shape {arr.shape}"
            )
        return arr.copy()

    def _compute_coefficients(self):
        dt, dx = self.dt, self.dx

        # H update: Hy[k] += C_H[k] * (Ez[k+1] - Ez[k])
        self.C_H = dt / (mu0 * self.mu_r * dx)

        # E update with optional conductivity (Crank-Nicolson in conductivity):
        #   Ca = (1 - σΔt/(2ε)) / (1 + σΔt/(2ε))
        #   Cb = (Δt/εΔx)       / (1 + σΔt/(2ε))
        denom = 1.0 + self.sigma_e * dt / (2.0 * eps0 * self.eps_r)
        self.Ca = (1.0 - self.sigma_e * dt / (2.0 * eps0 * self.eps_r)) / denom
        self.Cb = (dt / (eps0 * self.eps_r * dx)) / denom

        # Mur ABC coefficients using the local wave speed at each boundary
        c_left = c0 / np.sqrt(self.eps_r[0] * self.mu_r[0])
        c_right = c0 / np.sqrt(self.eps_r[-1] * self.mu_r[-1])
        S_left = c_left * dt / dx
        S_right = c_right * dt / dx
        self._mur_left = (S_left - 1.0) / (S_left + 1.0)
        self._mur_right = (S_right - 1.0) / (S_right + 1.0)

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def add_source(self, pos: int, func, source_type: str = "soft"):
        """Register a source at grid position *pos*.

        Parameters
        ----------
        pos : int
            Ez node index where the source is injected (0 ≤ pos ≤ N).
        func : callable
            Source waveform ``f(t) -> float`` where *t* is in seconds.
        source_type : {'soft', 'hard'}
            *'soft'* (default) adds to the field; *'hard'* overwrites it.
        """
        if not (0 <= pos <= self.N):
            raise ValueError(f"Source position {pos} out of range [0, {self.N}]")
        self._sources.append({"pos": pos, "func": func, "type": source_type})

    def step(self):
        """Advance the simulation by one Δt time step (Yee leapfrog)."""
        self._update_H()
        self._update_E()
        self.n += 1

    def run(self, n_steps: int):
        """Run the simulation for *n_steps* time steps.

        Parameters
        ----------
        n_steps : int
            Number of Yee leapfrog cycles to execute.
        """
        for _ in range(n_steps):
            self.step()

    def run_and_record(
        self,
        n_steps: int,
        record_interval: int = 1,
        probe_positions=None,
    ):
        """Run the simulation and optionally record field data.

        Parameters
        ----------
        n_steps : int
            Number of time steps to run.
        record_interval : int, optional
            Store a full Ez snapshot every *record_interval* steps.
        probe_positions : list of int, optional
            Ez node indices at which to record a time-domain signal.

        Returns
        -------
        snapshots : list of ndarray
            Ez field snapshots (recorded every *record_interval* steps).
        time_series : dict[int, list[float]]
            Mapping from probe position to list of recorded Ez values.
        """
        probe_positions = probe_positions or []
        snapshots: list[np.ndarray] = []
        time_series: dict[int, list] = {p: [] for p in probe_positions}

        for i in range(n_steps):
            self.step()
            if i % record_interval == 0:
                snapshots.append(self.Ez.copy())
                for p in probe_positions:
                    time_series[p].append(float(self.Ez[p]))

        return snapshots, time_series

    # ---------------------------------------------------------------------- #
    # Convenience properties
    # ---------------------------------------------------------------------- #

    @property
    def x_Ez(self) -> np.ndarray:
        """x-coordinates [m] of Ez nodes."""
        return np.arange(self.N + 1) * self.dx

    @property
    def x_Hy(self) -> np.ndarray:
        """x-coordinates [m] of Hy nodes (offset by dx/2)."""
        return (np.arange(self.N) + 0.5) * self.dx

    @property
    def time(self) -> float:
        """Current simulation time [s]."""
        return self.n * self.dt

    # ---------------------------------------------------------------------- #
    # Visualisation
    # ---------------------------------------------------------------------- #

    def plot_fields(self, ax=None, show_Hy: bool = True):
        """Plot the current Ez (and optionally Hy) field.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes (created if *None*).
        show_Hy : bool, optional
            If True also plot Hy scaled by η₀ to share the same axis.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.x_Ez, self.Ez, "b-", lw=1.5, label=r"$E_z$ [V/m]")
        if show_Hy:
            from .constants import eta0
            ax.plot(
                self.x_Hy,
                self.Hy * eta0,
                "r--",
                lw=1.5,
                label=r"$\eta_0 H_y$ [V/m]",
            )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("Field amplitude")
        ax.set_title(
            f"1-D FDTD  –  t = {self.time:.3e} s  (step {self.n})"
        )
        ax.legend()
        ax.grid(True, ls=":")
        return ax

    # ---------------------------------------------------------------------- #
    # Private field-update methods
    # ---------------------------------------------------------------------- #

    def _update_H(self):
        """Half-step: update Hy from Ez."""
        # Hy[k] is between Ez[k] and Ez[k+1]
        self.Hy += self.C_H * (self.Ez[1:] - self.Ez[:-1])

    def _update_E(self):
        """Full step: update Ez from Hy, apply BC and sources."""
        # Save boundary-neighbour values needed for Mur ABC
        ez_left_save = float(self.Ez[1])
        ez_right_save = float(self.Ez[-2])  # Ez[N-1]

        # ------- interior update -------
        # Ez[k] uses Hy[k] (right) and Hy[k-1] (left)
        self.Ez[1:-1] = (
            self.Ca[1:-1] * self.Ez[1:-1]
            + self.Cb[1:-1] * (self.Hy[1:] - self.Hy[:-1])
        )

        # ------- boundary conditions -------
        if self.boundary == "mur":
            # 1st-order Mur ABC:
            #   Ez[0]^{n+1} = Ez[1]^n + coeff*(Ez[1]^{n+1} - Ez[0]^n)
            new_left = ez_left_save + self._mur_left * (self.Ez[1] - self.Ez[0])
            new_right = ez_right_save + self._mur_right * (self.Ez[-2] - self.Ez[-1])
            self.Ez[0] = new_left
            self.Ez[-1] = new_right
        else:  # 'pec'
            self.Ez[0] = 0.0
            self.Ez[-1] = 0.0

        # Update stored values for the next call
        self._ez_left_old = ez_left_save
        self._ez_right_old = ez_right_save

        # ------- inject sources at new time (n+1) -------
        t_new = (self.n + 1) * self.dt
        for src in self._sources:
            val = float(src["func"](t_new))
            if src["type"] == "soft":
                self.Ez[src["pos"]] += val
            else:
                self.Ez[src["pos"]] = val
