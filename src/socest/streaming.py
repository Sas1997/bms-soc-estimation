"""Sample-by-sample (streaming) versions of the estimators, for edge deployment.

Each object has step(voltage, current, temp) -> SoC and is fed one 1 Hz sample
at a time, as a BMS would be. They produce the same numbers as the batch
estimators in socest.estimators (checked in tests/test_streaming.py).
"""

from __future__ import annotations

import numpy as np

from .cell import load_model
from .estimators import EKFEstimator, NeuralEstimator
from .kalman import EKF


class StreamingCoulomb:
    def __init__(self, capacity_ah: float, soc0: float = 1.0, dt: float = 1.0):
        self.k = dt / (3600.0 * capacity_ah)
        self.soc = soc0

    def step(self, voltage, current, temp):
        out = self.soc
        self.soc -= current * self.k
        return out


class StreamingEKF:
    def __init__(self, est: EKFEstimator, temp0: float, soc0: float = 1.0, dt: float = 1.0):
        self.est, self.dt, self.t_amb = est, dt, float(temp0)
        self.model = load_model(est.variant, capacity_ah=est.cap_of(temp0))
        n = self.model.n_states
        x0 = np.zeros(n)
        x0[0] = soc0
        self.ekf = EKF(self.model, x0, np.diag([est.p0_soc**2] + [est.p0_v**2] * (n - 1)),
                       np.diag([est.q_soc**2] + [est.q_v**2] * (n - 1)), 1.0)

    def step(self, voltage, current, temp):
        t_eff = temp + self.model.thermal_gain * (temp - self.t_amb)
        r = float((self.est.k_r * self.est.sigma_v(temp)) ** 2)
        self.ekf.update(voltage, current, t_eff, r)
        soc = float(self.ekf.x[0])
        self.ekf.predict(current, self.dt, t_eff)
        return soc

    @property
    def soc_std(self) -> float:
        return float(np.sqrt(max(self.ekf.P[0, 0], 0.0)))


class StreamingNN:
    """Keeps the last `window` normalised samples in a buffer; one forward pass per step."""

    def __init__(self, est: NeuralEstimator):
        import torch

        self.torch = torch
        self.model, self.norm, self.window = est.model, est.norm, est.meta["window"]
        self.buf = None

    def step(self, voltage, current, temp):
        x = ((np.array([voltage, current, temp]) - self.norm.mean) / self.norm.std).astype(np.float32)
        if self.buf is None:
            self.buf = np.repeat(x[None], self.window, axis=0)   # start-up: pad with the first sample
        else:
            self.buf[:-1] = self.buf[1:]
            self.buf[-1] = x
        with self.torch.no_grad():
            return float(self.model(self.torch.from_numpy(self.buf[None]))[0])
