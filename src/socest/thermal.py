"""Effective (core) temperature from the measured cell temperature.

The dataset reports one temperature per sample. Under load the inside of the
cell is warmer than its surface. For internal and external thermal resistances
in series, the steady-state core temperature is

    T_core = T_surface + g * (T_surface - T_ambient),   g = R_internal / R_external.

T_ambient is taken as the rested temperature at the start of the record. g = 0
returns the measured temperature unchanged. g is a calibration choice made on
training data (experiments/m2_calibrate_slow.py), not a measured property.
"""

from __future__ import annotations

import numpy as np


def effective_temperature(temp, gain: float, t_ambient: float | None = None, n_start: int = 30):
    temp = np.asarray(temp, dtype=float)
    if not gain:
        return temp
    t0 = float(np.mean(temp[:n_start])) if t_ambient is None else t_ambient
    return temp + gain * (temp - t0)
