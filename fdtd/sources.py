"""Source waveforms for FDTD simulations.

All functions follow the signature ``f(t)`` where *t* is the simulation
time in seconds and the return value is dimensionless (it will be scaled
by the FDTD solver before injection).
"""

import numpy as np


def gaussian_pulse(t0, tau):
    """Return a Gaussian pulse source function.

    Parameters
    ----------
    t0 : float
        Peak time [s].
    tau : float
        Pulse half-width (standard deviation) [s].

    Returns
    -------
    callable
        ``f(t) = exp(-((t - t0) / tau)^2)``
    """
    def f(t):
        return np.exp(-((t - t0) / tau) ** 2)
    f.__name__ = "gaussian_pulse"
    return f


def differentiated_gaussian(t0, tau):
    """Return a differentiated Gaussian source function (odd symmetry).

    This is the first time-derivative of the Gaussian pulse.  It produces
    a zero-mean waveform that is well-suited for broadband excitation.

    Parameters
    ----------
    t0 : float
        Zero-crossing time (inflection of the parent Gaussian) [s].
    tau : float
        Pulse half-width [s].

    Returns
    -------
    callable
        ``f(t) = -2*(t-t0)/tau^2 * exp(-((t-t0)/tau)^2)``
    """
    def f(t):
        return -2.0 * (t - t0) / tau ** 2 * np.exp(-((t - t0) / tau) ** 2)
    f.__name__ = "differentiated_gaussian"
    return f


def ricker_wavelet(f0):
    """Return a Ricker wavelet (second derivative of Gaussian) source.

    The Ricker wavelet is widely used in FDTD because it has a narrow
    frequency band centred at *f0* and is zero-mean.

    Parameters
    ----------
    f0 : float
        Dominant (peak) frequency [Hz].

    Returns
    -------
    callable
        Ricker wavelet function of time.
    """
    tau = np.sqrt(6.0) / (np.pi * f0)
    t0 = tau  # shift so the pulse starts near zero

    def f(t):
        u = np.pi * f0 * (t - t0)
        return (1.0 - 2.0 * u ** 2) * np.exp(-(u ** 2))
    f.__name__ = "ricker_wavelet"
    return f


def sinusoidal(f0, amplitude=1.0, phase=0.0):
    """Return a continuous-wave sinusoidal source.

    Parameters
    ----------
    f0 : float
        Frequency [Hz].
    amplitude : float, optional
        Peak amplitude (default 1.0).
    phase : float, optional
        Initial phase [rad] (default 0.0).

    Returns
    -------
    callable
        ``f(t) = amplitude * sin(2*pi*f0*t + phase)``
    """
    def f(t):
        return amplitude * np.sin(2.0 * np.pi * f0 * t + phase)
    f.__name__ = "sinusoidal"
    return f


def ramped_sinusoidal(f0, t_ramp, amplitude=1.0):
    """Return a sinusoidal source with a linear amplitude ramp.

    The amplitude grows linearly from 0 to *amplitude* over [0, t_ramp]
    and remains constant afterwards.

    Parameters
    ----------
    f0 : float
        Frequency [Hz].
    t_ramp : float
        Ramp duration [s].
    amplitude : float, optional
        Final amplitude (default 1.0).

    Returns
    -------
    callable
    """
    def f(t):
        ramp = min(t / t_ramp, 1.0) if t_ramp > 0.0 else 1.0
        return amplitude * ramp * np.sin(2.0 * np.pi * f0 * t)
    f.__name__ = "ramped_sinusoidal"
    return f
