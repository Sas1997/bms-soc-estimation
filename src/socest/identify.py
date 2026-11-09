"""Equivalent circuit parameter identification from pulse (HPPC) segments.

Method: for one segment (rest, current pulse, relaxation), the OCV is anchored to
the measured rest voltage just before the pulse, SoC follows coulomb counting, and
R0, R_i, tau_i are found by bounded nonlinear least squares on the terminal
voltage (scipy.optimize.least_squares, trust-region reflective). Parameters are
optimised in log space so they stay positive. The RC time constants are kept in
separate, ordered ranges so branch identities are stable across segments.

Fit quality is reported as voltage RMSE over the segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

from .ecm import ECMParams
from .ocv import OCVCurve

# Default time-constant search ranges (s) per number of RC branches.
DEFAULT_TAU_BOUNDS = {
    1: [(0.5, 2000.0)],
    2: [(0.5, 60.0), (60.0, 3000.0)],
}


def rc_voltages(current: np.ndarray, dt: np.ndarray, r: float, tau: float) -> np.ndarray:
    """Voltage across one RC branch (ZOH, starting from 0 V). Output at sample k
    is the state before applying current[k], consistent with TheveninModel."""
    n = current.size
    out = np.zeros(n)
    if n < 2:
        return out
    if np.allclose(dt, dt[0]):
        a = np.exp(-dt[0] / tau)
        # x[k+1] = a x[k] + r (1-a) i[k]  ->  lfilter over i, then shift by one sample
        y = lfilter([r * (1.0 - a)], [1.0, -a], current[:-1])
        out[1:] = y
        return out
    a = np.exp(-dt / tau)
    b = r * (1.0 - a)
    x = 0.0
    for k in range(n - 1):
        x = a[k] * x + b[k] * current[k]
        out[k + 1] = x
    return out


@dataclass
class SegmentFit:
    params: ECMParams
    rmse_v: float
    soc_mid: float
    temp: float
    success: bool


def fit_segment(
    time,
    current,
    voltage,
    soc0: float,
    capacity_ah: float,
    ocv: OCVCurve,
    temp: float,
    n_rc: int = 2,
    tau_bounds=None,
) -> SegmentFit:
    """Fit an n-RC model to one segment that starts at rest.

    time    : (N,) s
    current : (N,) A, i > 0 discharge
    voltage : (N,) V
    soc0    : SoC at the first sample
    """
    t = np.asarray(time, dtype=float)
    i = np.asarray(current, dtype=float)
    v = np.asarray(voltage, dtype=float)
    dt = np.diff(t, append=t[-1] + (t[-1] - t[-2]))
    soc = soc0 - np.concatenate(([0.0], np.cumsum(i[:-1] * dt[:-1]))) / (3600.0 * capacity_ah)
    # Anchor OCV to the pre-pulse rest voltage (removes residual hysteresis offset).
    ocv_t = ocv(soc, temp)
    ocv_t = ocv_t + (v[0] - ocv_t[0])

    tau_bounds = tau_bounds or DEFAULT_TAU_BOUNDS[n_rc]

    # Initial guesses: R0 from the voltage step at the first current edge.
    edge = int(np.argmax(np.abs(i) > 0.05 * np.max(np.abs(i))))
    di = i[edge] - i[max(edge - 1, 0)]
    dv = v[edge] - v[max(edge - 1, 0)]
    r0_guess = float(np.clip(-dv / di if abs(di) > 1e-6 else 0.02, 1e-4, 0.5))
    x0 = [np.log(r0_guess)]
    lo, hi = [np.log(1e-5)], [np.log(1.0)]
    for tb in tau_bounds:
        x0 += [np.log(0.5 * r0_guess), np.log(np.sqrt(tb[0] * tb[1]))]
        lo += [np.log(1e-6), np.log(tb[0])]
        hi += [np.log(1.0), np.log(tb[1])]

    def unpack(x):
        e = np.exp(x)
        return e[0], e[1::2], e[2::2]

    def model_v(x):
        r0, rs, taus = unpack(x)
        vrc = sum(rc_voltages(i, dt, r, tau) for r, tau in zip(rs, taus))
        return ocv_t - vrc - r0 * i

    res = least_squares(lambda x: model_v(x) - v, x0, bounds=(lo, hi), method="trf", x_scale="jac")
    r0, rs, taus = unpack(res.x)
    rmse = float(np.sqrt(np.mean(res.fun**2)))
    return SegmentFit(
        params=ECMParams(r0=float(r0), r=tuple(map(float, rs)), tau=tuple(map(float, taus))),
        rmse_v=rmse,
        soc_mid=float(np.mean(soc)),
        temp=float(temp),
        success=bool(res.success),
    )
