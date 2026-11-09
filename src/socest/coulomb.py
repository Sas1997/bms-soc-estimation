"""Open-loop coulomb counting.

Same discretisation as TheveninModel.transition, so the two agree exactly.
Current convention: i > 0 is discharge.
"""

from __future__ import annotations

import numpy as np


def coulomb_count(current, dt, soc0: float, capacity_ah: float, eta_charge: float = 1.0):
    """SoC trajectory from integrated current.

    current : array (N,), A
    dt      : scalar or array (N,), s; dt[k] is the time from sample k to k+1
    soc0    : SoC at sample 0 (fraction, not percent)
    Returns an array (N,) with the SoC at each sample, starting at soc0.
    """
    current = np.asarray(current, dtype=float)
    dt = np.broadcast_to(np.asarray(dt, dtype=float), current.shape)
    eta = np.where(current < 0, eta_charge, 1.0)
    dsoc = -eta * current * dt / (3600.0 * capacity_ah)
    soc = np.empty_like(current)
    soc[0] = soc0
    soc[1:] = soc0 + np.cumsum(dsoc[:-1])
    return soc
