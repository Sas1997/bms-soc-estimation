"""Resample logged data onto a uniform grid for estimation.

Each output sample k covers the bin [t0 + k*dt, t0 + (k+1)*dt). Current is the
time-weighted mean over the bin, so charge is preserved exactly (the tester
logs at about 0.1 s, so each 1 s bin holds about 10 raw samples held with ZOH).
Voltage and temperature are bin means. The reference SoC is taken at the bin
start, which matches the estimator convention: SoC[k] is the state before
applying current[k] over the bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_uniform(df: pd.DataFrame, dt: float = 1.0) -> pd.DataFrame:
    t = df["time_s"].to_numpy(dtype=float)
    t0 = t[0]
    n = int(np.floor((t[-1] - t0) / dt))
    if n < 1:
        raise ValueError("record shorter than one bin")
    edges = t0 + dt * np.arange(n + 1)

    # Each raw sample j holds over [t_j, t_{j+1}); integrate its overlap with every bin.
    seg_t0 = t[:-1]
    seg_t1 = t[1:]
    out = {"time_s": edges[:-1] - t0}
    for col, raw in (("current_a", "current_a"), ("voltage_v", "voltage_v"), ("temp_degc", "temp_degc")):
        y = df[raw].to_numpy(dtype=float)[:-1]
        # cumulative integral of the ZOH signal, evaluated at bin edges
        cum = np.concatenate(([0.0], np.cumsum(y * (seg_t1 - seg_t0))))
        k = np.searchsorted(seg_t1, edges, side="left")  # segment containing each edge
        k = np.clip(k, 0, len(seg_t0) - 1)
        F = cum[k] + y[k] * (np.clip(edges, seg_t0[k], seg_t1[k]) - seg_t0[k])
        out[col] = np.diff(F) / dt
    if "soc_ref" in df:
        out["soc_ref"] = np.interp(edges[:-1], t, df["soc_ref"].to_numpy())
    return pd.DataFrame(out)
