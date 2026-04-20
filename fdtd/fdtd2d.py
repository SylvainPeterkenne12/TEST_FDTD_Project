"""2D FDTD simulation using the Yee algorithm (TMz mode).

The TMz (transverse-magnetic in z) formulation retains three field
components:

* **Ez** (i, j)     – at grid point  (i·Δx,  j·Δy)
* **Hx** (i, j)     – at grid point  (i·Δx, (j+½)·Δy)
* **Hy** (i, j)     – at grid point ((i+½)·Δx,  j·Δy)

Grid layout (Nx × Ny cells, with (Nx+1)×(Ny+1) Ez nodes)::

          j+1  ·····Ez[i,j+1]·········Ez[i+1,j+1]···
                         |                  |
    Hx[i,j] is between   |    Hy[i,j] is   |
    Ez[i,j] & Ez[i,j+1]  |    between       |
                         |    Ez[i,j] &     |
           j  ·····Ez[i,j]·····Hy[i,j]·····Ez[i+1,j]···
                    i                  i+1

Yee update equations per time step:

    Hx[i,j] -= (Δt / μ₀μᵣΔy) * (Ez[i,j+1] − Ez[i,j])
    Hy[i,j] += (Δt / μ₀μᵣΔx) * (Ez[i+1,j] − Ez[i,j])
    Ez[i,j] += (Δt / ε₀εᵣ) * ((Hy[i,j]−Hy[i-1,j])/Δx
                                − (Hx[i,j]−Hx[i,j-1])/Δy)

Absorbing boundary: first-order Mur ABC on all four sides.
Stability: Courant number S = c₀Δt/Δx ≤ 1/√2 (for Δx = Δy).
"""

import numpy as np
import matplotlib.pyplot as plt

from .constants import c0, eps0, mu0


class FDTD2D:
    """2D FDTD simulation (TMz) using the Yee staggered-grid algorithm.

    Parameters
    ----------
    Nx, Ny : int
        Number of cells in x and y directions.
    dx, dy : float
        Cell sizes [m].  *dy* defaults to *dx* if not given.
    eps_r : float or ndarray, shape (Nx+1, Ny+1), optional
        Relative permittivity at Ez nodes (default: 1.0).
    mu_r : float or ndarray, shape (Nx+1, Ny+1), optional
        Relative permeability used for both Hx and Hy (default: 1.0).
    sigma_e : float or ndarray, shape (Nx+1, Ny+1), optional
        Electric conductivity [S/m] at Ez nodes (default: 0.0).
    courant : float, optional
        Courant number S = c₀Δt/Δx.  Must satisfy S ≤ 1/√2 for Δx=Δy.
        Default: ``0.99/sqrt(2)``.
    boundary : {'mur', 'pec'}
        Boundary condition (default: 'mur').

    Attributes
    ----------
    Ez : ndarray, shape (Nx+1, Ny+1)
    Hx : ndarray, shape (Nx+1, Ny)
    Hy : ndarray, shape (Nx, Ny+1)
    n  : int
        Current time-step index.
    dt : float
        Time step [s].
    """

    def __init__(
        self,
        Nx: int,
        Ny: int,
        dx: float,
        dy: float = None,
        eps_r=1.0,
        mu_r=1.0,
        sigma_e=0.0,
        courant: float = 0.99 / np.sqrt(2.0),
        boundary: str = "mur",
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.courant = courant
        self.boundary = boundary

        # Time step
        self.dt = courant * dx / c0

        # ------------------------------------------------------------------ #
        # Material arrays – all referenced to Ez grid (Nx+1) × (Ny+1)
        # ------------------------------------------------------------------ #
        shape_E = (Nx + 1, Ny + 1)
        self.eps_r = self._broadcast(eps_r, shape_E)
        self.mu_r = self._broadcast(mu_r, shape_E)
        self.sigma_e = self._broadcast(sigma_e, shape_E)

        # ------------------------------------------------------------------ #
        # Update coefficients
        # ------------------------------------------------------------------ #
        self._compute_coefficients()

        # ------------------------------------------------------------------ #
        # Field arrays
        # ------------------------------------------------------------------ #
        self.Ez = np.zeros((Nx + 1, Ny + 1), dtype=float)
        self.Hx = np.zeros((Nx + 1, Ny), dtype=float)   # at (i, j+½)
        self.Hy = np.zeros((Nx, Ny + 1), dtype=float)   # at (i+½, j)

        # ------------------------------------------------------------------ #
        # Mur ABC storage (boundary slices at previous time step)
        # ------------------------------------------------------------------ #
        self._ez_xlo_old = np.zeros(Ny + 1)   # Ez[1,:]  at previous step
        self._ez_xhi_old = np.zeros(Ny + 1)   # Ez[Nx-1,:] at previous step
        self._ez_ylo_old = np.zeros(Nx + 1)   # Ez[:,1]  at previous step
        self._ez_yhi_old = np.zeros(Nx + 1)   # Ez[:,Ny-1] at previous step

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
    def _broadcast(val, shape: tuple) -> np.ndarray:
        if np.isscalar(val):
            return float(val) * np.ones(shape)
        arr = np.asarray(val, dtype=float)
        if arr.shape != shape:
            raise ValueError(
                f"Expected scalar or array of shape {shape}, got {arr.shape}"
            )
        return arr.copy()

    def _compute_coefficients(self):
        dt, dx, dy = self.dt, self.dx, self.dy

        # ---- H update coefficients ----
        # For Hx (at half y-index): use mu_r averaged between j and j+1
        mu_hx = 0.5 * (self.mu_r[:, :-1] + self.mu_r[:, 1:])  # (Nx+1, Ny)
        self.C_Hx = dt / (mu0 * mu_hx * dy)

        # For Hy (at half x-index): use mu_r averaged between i and i+1
        mu_hy = 0.5 * (self.mu_r[:-1, :] + self.mu_r[1:, :])  # (Nx, Ny+1)
        self.C_Hy = dt / (mu0 * mu_hy * dx)

        # ---- E update coefficients (Crank-Nicolson for conductivity) ----
        denom = 1.0 + self.sigma_e * dt / (2.0 * eps0 * self.eps_r)
        self.Ca = (1.0 - self.sigma_e * dt / (2.0 * eps0 * self.eps_r)) / denom
        self.Cb_x = (dt / (eps0 * self.eps_r * dx)) / denom
        self.Cb_y = (dt / (eps0 * self.eps_r * dy)) / denom

        # ---- Mur ABC coefficients ----
        # Use vacuum (or average boundary cell) speed; for simplicity use c0.
        S_x = c0 * dt / dx
        S_y = c0 * dt / dy
        self._mur_x = (S_x - 1.0) / (S_x + 1.0)
        self._mur_y = (S_y - 1.0) / (S_y + 1.0)

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def add_source(self, ix: int, iy: int, func, source_type: str = "soft"):
        """Register a point source at Ez node (ix, iy).

        Parameters
        ----------
        ix, iy : int
            Node indices (0 ≤ ix ≤ Nx, 0 ≤ iy ≤ Ny).
        func : callable
            Source waveform ``f(t) -> float``.
        source_type : {'soft', 'hard'}
            'soft' (default) adds to the field; 'hard' overwrites it.
        """
        if not (0 <= ix <= self.Nx and 0 <= iy <= self.Ny):
            raise ValueError(
                f"Source position ({ix},{iy}) out of range "
                f"[(0,0), ({self.Nx},{self.Ny})]"
            )
        self._sources.append({"ix": ix, "iy": iy, "func": func, "type": source_type})

    def step(self):
        """Advance one Δt time step."""
        self._update_H()
        self._update_E()
        self.n += 1

    def run(self, n_steps: int):
        """Run for *n_steps* time steps."""
        for _ in range(n_steps):
            self.step()

    def run_and_record(
        self,
        n_steps: int,
        record_interval: int = 1,
        probe_positions=None,
    ):
        """Run and record Ez snapshots and optional probe signals.

        Parameters
        ----------
        n_steps : int
        record_interval : int
            Snapshot every *record_interval* steps.
        probe_positions : list of (int, int), optional
            (ix, iy) pairs to record time-domain Ez signals.

        Returns
        -------
        snapshots : list of ndarray, shape (Nx+1, Ny+1)
        time_series : dict[(int,int), list[float]]
        """
        probe_positions = probe_positions or []
        snapshots: list[np.ndarray] = []
        time_series: dict = {p: [] for p in probe_positions}

        for i in range(n_steps):
            self.step()
            if i % record_interval == 0:
                snapshots.append(self.Ez.copy())
                for p in probe_positions:
                    time_series[p].append(float(self.Ez[p[0], p[1]]))

        return snapshots, time_series

    # ---------------------------------------------------------------------- #
    # Convenience properties
    # ---------------------------------------------------------------------- #

    @property
    def x(self) -> np.ndarray:
        """x-axis coordinates [m] of Ez nodes."""
        return np.arange(self.Nx + 1) * self.dx

    @property
    def y(self) -> np.ndarray:
        """y-axis coordinates [m] of Ez nodes."""
        return np.arange(self.Ny + 1) * self.dy

    @property
    def time(self) -> float:
        """Current simulation time [s]."""
        return self.n * self.dt

    # ---------------------------------------------------------------------- #
    # Visualisation
    # ---------------------------------------------------------------------- #

    def plot_Ez(self, ax=None, vmax=None, cmap="RdBu_r", title=None):
        """Plot the Ez field as a 2-D colour map.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        vmax : float, optional
            Colour scale limit.  Defaults to the maximum absolute value.
        cmap : str, optional
            Matplotlib colour map name (default: 'RdBu_r').
        title : str, optional

        Returns
        -------
        fig, ax, im
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.get_figure()

        XX, YY = np.meshgrid(self.x, self.y, indexing="ij")
        if vmax is None:
            vmax = np.abs(self.Ez).max() or 1.0
        im = ax.pcolormesh(
            XX, YY, self.Ez,
            cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto",
        )
        fig.colorbar(im, ax=ax, label=r"$E_z$ [V/m]")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(
            title or f"2-D FDTD  –  t = {self.time:.3e} s  (step {self.n})"
        )
        ax.set_aspect("equal")
        return fig, ax, im

    # ---------------------------------------------------------------------- #
    # Private field-update methods
    # ---------------------------------------------------------------------- #

    def _update_H(self):
        """Half-step: update Hx and Hy from Ez."""
        # Hx[i,j] is between Ez[i,j] and Ez[i,j+1]
        self.Hx -= self.C_Hx * (self.Ez[:, 1:] - self.Ez[:, :-1])

        # Hy[i,j] is between Ez[i,j] and Ez[i+1,j]
        self.Hy += self.C_Hy * (self.Ez[1:, :] - self.Ez[:-1, :])

    def _update_E(self):
        """Full step: update Ez from Hx/Hy, apply BC and sources."""
        # Save boundary-neighbour slices needed for Mur ABC
        ez_xlo_save = self.Ez[1, :].copy()
        ez_xhi_save = self.Ez[-2, :].copy()
        ez_ylo_save = self.Ez[:, 1].copy()
        ez_yhi_save = self.Ez[:, -2].copy()

        # ------- interior update: Ez[i,j] for i=1..Nx-1, j=1..Ny-1 -------
        i_sl = slice(1, self.Nx)   # i = 1 .. Nx-1
        j_sl = slice(1, self.Ny)   # j = 1 .. Ny-1

        # ∂Hy/∂x  at (i,j): (Hy[i,j] − Hy[i-1,j]) / Δx
        #   Hy shape (Nx, Ny+1); Hy[k,j] is at ((k+½)Δx, jΔy)
        # ∂Hx/∂y  at (i,j): (Hx[i,j] − Hx[i,j-1]) / Δy
        #   Hx shape (Nx+1, Ny); Hx[i,k] is at (iΔx, (k+½)Δy)
        self.Ez[i_sl, j_sl] = (
            self.Ca[i_sl, j_sl] * self.Ez[i_sl, j_sl]
            + self.Cb_x[i_sl, j_sl] * (self.Hy[1:self.Nx, j_sl] - self.Hy[0:self.Nx - 1, j_sl])
            - self.Cb_y[i_sl, j_sl] * (self.Hx[i_sl, 1:self.Ny] - self.Hx[i_sl, 0:self.Ny - 1])
        )

        # ------- boundary conditions -------
        if self.boundary == "mur":
            mx, my = self._mur_x, self._mur_y
            # x = 0  (left wall)
            self.Ez[0, :] = ez_xlo_save + mx * (self.Ez[1, :] - self.Ez[0, :])
            # x = Nx (right wall)
            self.Ez[-1, :] = ez_xhi_save + mx * (self.Ez[-2, :] - self.Ez[-1, :])
            # y = 0  (bottom wall)
            self.Ez[:, 0] = ez_ylo_save + my * (self.Ez[:, 1] - self.Ez[:, 0])
            # y = Ny (top wall)
            self.Ez[:, -1] = ez_yhi_save + my * (self.Ez[:, -2] - self.Ez[:, -1])
        else:  # 'pec'
            self.Ez[0, :] = 0.0
            self.Ez[-1, :] = 0.0
            self.Ez[:, 0] = 0.0
            self.Ez[:, -1] = 0.0

        # Update stored boundary slices
        self._ez_xlo_old = ez_xlo_save
        self._ez_xhi_old = ez_xhi_save
        self._ez_ylo_old = ez_ylo_save
        self._ez_yhi_old = ez_yhi_save

        # ------- inject sources at new time (n+1) -------
        t_new = (self.n + 1) * self.dt
        for src in self._sources:
            val = float(src["func"](t_new))
            ix, iy = src["ix"], src["iy"]
            if src["type"] == "soft":
                self.Ez[ix, iy] += val
            else:
                self.Ez[ix, iy] = val
