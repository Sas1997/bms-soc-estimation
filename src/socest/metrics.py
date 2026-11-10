"""SoC error metrics and the sensor perturbations used in the robustness tests.

All SoC values are fractions in [0, 1]; metrics are reported in percentage points
(multiply by 100) by the experiment scripts, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def soc_errors(soc_est, soc_ref, skip: int = 0) -> dict:
    """RMSE, MAE and maximum absolute error, ignoring the first ``skip`` samples.

    ``skip`` lets a filter's initial convergence be excluded explicitly; the
    experiments report both skip=0 and a stated convergence window.
    """
    e = np.asarray(soc_est, dtype=float)[skip:] - np.asarray(soc_ref, dtype=float)[skip:]
    if e.size == 0:
        raise ValueError("no samples left after skip")
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "max_abs": float(np.max(np.abs(e))),
    }


def convergence_time(soc_est, soc_ref, time, band: float = 0.02):
    """First time (s, relative to time[0]) after which |error| stays within ``band``.
    Returns None if it never settles."""
    err = np.abs(np.asarray(soc_est) - np.asarray(soc_ref))
    outside = np.nonzero(err > band)[0]
    if outside.size == 0:
        return 0.0
    k = outside[-1] + 1
    if k >= err.size:
        return None
    return float(time[k] - time[0])


@dataclass(frozen=True)
class Perturbation:
    """Sensor corruption applied to measurements before estimation.

    v_noise_std  : V, white Gaussian noise on voltage
    i_noise_std  : A, white Gaussian noise on current
    i_bias       : A, constant offset added to measured current
    t_noise_std  : degC, white Gaussian noise on temperature
    soc0_error   : SoC fraction added to the estimator's initial SoC (not to data)
    start_s      : estimation starts this many seconds into the record (the cell
                   is then under load and not at rest), 0 = from the beginning
    """

    name: str = "nominal"
    v_noise_std: float = 0.0
    i_noise_std: float = 0.0
    i_bias: float = 0.0
    t_noise_std: float = 0.0
    soc0_error: float = 0.0
    start_s: int = 0

    def apply(self, voltage, current, temp, seed: int = 0):
        rng = np.random.default_rng(seed)
        v = np.asarray(voltage, dtype=float)
        i = np.asarray(current, dtype=float)
        t = np.asarray(temp, dtype=float)
        if self.v_noise_std:
            v = v + rng.normal(0.0, self.v_noise_std, v.shape)
        if self.i_noise_std:
            i = i + rng.normal(0.0, self.i_noise_std, i.shape)
        if self.i_bias:
            i = i + self.i_bias
        if self.t_noise_std:
            t = t + rng.normal(0.0, self.t_noise_std, t.shape)
        return v, i, t
