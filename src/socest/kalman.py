"""Extended Kalman filter for SoC on an n-RC Thevenin model.

Formulation follows Plett, "Extended Kalman filtering for battery management
systems of LiPB-based HEV battery packs, Part 3", J. Power Sources 134 (2004)
277-292: the state is [soc, v_1..v_n], the measurement is terminal voltage.

Order per sample k (current i[k] held constant over [t_k, t_k+1)):
    1. measurement update with v[k]
    2. time update to k+1 with i[k]
The estimate returned for sample k is the posterior after step 1.
The covariance update uses the Joseph form for numerical robustness.
"""

from __future__ import annotations

import numpy as np

from .ecm import TheveninModel


class EKF:
    def __init__(self, model: TheveninModel, x0, P0, Q, R: float):
        """
        model : TheveninModel
        x0    : initial state [soc, v_1..v_n]
        P0    : initial state covariance (n_states x n_states)
        Q     : process noise covariance per step (n_states x n_states)
        R     : voltage measurement noise variance, V^2
        """
        self.model = model
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.Q = np.asarray(Q, dtype=float)
        self.R = float(R)
        n = model.n_states
        if self.x.shape != (n,) or self.P.shape != (n, n) or self.Q.shape != (n, n):
            raise ValueError("x0, P0, Q dimensions must match the model state size")
        self._I = np.eye(n)

    def update(self, v_meas: float, i: float, temp: float):
        """Measurement update. Returns the innovation (V)."""
        v_pred, H = self.model.output(self.x, i, temp)
        innov = v_meas - v_pred
        PHt = self.P @ H
        S = float(H @ PHt) + self.R
        K = PHt / S
        self.x = self.x + K * innov
        IKH = self._I - np.outer(K, H)
        self.P = IKH @ self.P @ IKH.T + self.R * np.outer(K, K)
        return innov

    def predict(self, i: float, dt: float, temp: float):
        """Time update over one sample interval."""
        self.x, A = self.model.transition(self.x, i, dt, temp)
        self.P = A @ self.P @ A.T + self.Q

    def run(self, voltage, current, dt, temp):
        """Filter a full sequence.

        Returns dict with soc (N,), soc_std (N,), innovation (N,).
        """
        voltage = np.asarray(voltage, dtype=float)
        current = np.asarray(current, dtype=float)
        n = voltage.size
        dt = np.broadcast_to(np.asarray(dt, dtype=float), (n,))
        temp = np.broadcast_to(np.asarray(temp, dtype=float), (n,))
        soc = np.empty(n)
        std = np.empty(n)
        innov = np.empty(n)
        for k in range(n):
            innov[k] = self.update(voltage[k], current[k], temp[k])
            soc[k] = self.x[0]
            std[k] = np.sqrt(max(self.P[0, 0], 0.0))
            self.predict(current[k], dt[k], temp[k])
        return {"soc": soc, "soc_std": std, "innovation": innov}
