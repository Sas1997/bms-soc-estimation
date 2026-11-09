"""Thevenin equivalent circuit model with n parallel RC branches.

State vector x = [soc, v_1, ..., v_n], where v_i is the voltage across RC branch i.
Current convention: i > 0 is discharge (see socest.__init__).

Continuous-time model (Plett, Battery Management Systems Vol. 1, 2015, ch. 2):
    d soc / dt = -eta * i / (3600 * Q)
    d v_i / dt = -v_i / tau_i + i / C_i,      tau_i = R_i * C_i
    v_terminal = OCV(soc, T) - sum(v_i) - R0 * i

Discretised with a zero-order hold on current, which is exact for piecewise
constant current:
    v_i[k+1] = a_i * v_i[k] + R_i * (1 - a_i) * i[k],   a_i = exp(-dt / tau_i)

Optional charge-transfer term. At low temperature the pulse resistance of the
cell falls with current (see results/tables/m2_cold_diagnostics.md). This is
modelled with the Butler-Volmer equation for symmetric transfer coefficients
(Bard and Faulkner, Electrochemical Methods, 2nd ed., 2001, ch. 3), inverted
for the overpotential and treated as instantaneous:
    eta(i, T) = n * (2 R T / F) * asinh(i / (2 i0)),   T in kelvin
    v_terminal = OCV - sum(v_i) - R0 * i - eta(i, T)
i0 (A) is a lumped exchange current; i0 = inf switches the term off. n is the
number of charge-transfer interfaces in series. n = 1 is used: on the HPPC
pulses n = 2 fitted no better (16.6 vs 17.0 mV pooled RMSE at -20 degC).

Finding (results/tables/m2_ecm_selection.md): the term improves the pulse fits
in the cold but does not improve drive-cycle voltage prediction once the slow
branch is calibrated, so the selected model does not use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .ocv import OCVCurve


@dataclass(frozen=True)
class ECMParams:
    """Parameters of an n-RC Thevenin model at one operating point.

    r0  : series resistance, ohm
    r   : RC branch resistances, ohm
    tau : RC branch time constants, s
    i0  : lumped exchange current of the Butler-Volmer term, A (inf = off)
    bv_n: number of series charge-transfer interfaces in the Butler-Volmer term
    """

    r0: float
    r: Sequence[float]
    tau: Sequence[float]
    i0: float = float("inf")
    bv_n: int = 1

    def __post_init__(self):
        if len(self.r) != len(self.tau):
            raise ValueError("r and tau must have the same length")

    @property
    def n_rc(self) -> int:
        return len(self.r)


GAS_R = 8.314462618  # J / (mol K)
FARADAY = 96485.33212  # C / mol


def bv_overpotential(i, temp_c, i0, n=1):
    """Butler-Volmer overpotential (V) for current i (A), temperature (degC),
    exchange current i0 (A) and n series interfaces."""
    if not np.isfinite(i0):
        return 0.0 * np.asarray(i, dtype=float) if np.ndim(i) else 0.0
    return n * 2.0 * GAS_R * (np.asarray(temp_c, dtype=float) + 273.15) / FARADAY * np.arcsinh(np.asarray(i) / (2.0 * i0))


# A parameter provider returns ECMParams for a given (soc, temp) operating point.
ParamProvider = Callable[[float, float], ECMParams]


def constant_params(p: ECMParams) -> ParamProvider:
    """Provider that ignores SoC and temperature."""
    return lambda soc, temp: p


class TheveninModel:
    """Discrete-time n-RC Thevenin model."""

    def __init__(
        self,
        ocv: OCVCurve,
        params: ParamProvider | ECMParams,
        capacity_ah: float,
        eta_charge: float = 1.0,
    ):
        if capacity_ah <= 0:
            raise ValueError("capacity_ah must be positive")
        self.ocv = ocv
        self.params = constant_params(params) if isinstance(params, ECMParams) else params
        self.capacity_ah = float(capacity_ah)
        self.eta_charge = float(eta_charge)
        self.n_rc = self.params(0.5, 25.0).n_rc

    @property
    def n_states(self) -> int:
        return 1 + self.n_rc

    def _eta(self, i: float) -> float:
        return self.eta_charge if i < 0 else 1.0

    def transition(self, x: np.ndarray, i: float, dt: float, temp: float):
        """Return (x_next, A) where A = d x_next / d x.

        Parameter dependence on SoC is held fixed over the step, so A ignores
        d(params)/d(soc). This is the usual simplification for table-based ECMs.
        """
        p = self.params(float(x[0]), temp)
        a = np.exp(-dt / np.asarray(p.tau, dtype=float))
        x_next = np.empty_like(x)
        x_next[0] = x[0] - self._eta(i) * i * dt / (3600.0 * self.capacity_ah)
        x_next[1:] = a * x[1:] + np.asarray(p.r) * (1.0 - a) * i
        A = np.diag(np.concatenate(([1.0], a)))
        return x_next, A

    def output(self, x: np.ndarray, i: float, temp: float):
        """Return (v_terminal, H) where H = d v_terminal / d x."""
        p = self.params(float(x[0]), temp)
        v = float(self.ocv(x[0], temp)) - float(np.sum(x[1:])) - p.r0 * i - float(bv_overpotential(i, temp, p.i0, p.bv_n))
        H = np.empty(self.n_states)
        H[0] = float(self.ocv.dsoc(x[0], temp))
        H[1:] = -1.0
        return v, H

    def simulate(self, current, dt, temp, soc0: float, v_rc0=None):
        """Simulate terminal voltage for a current profile.

        current, dt, temp may be arrays of equal length or scalars (dt, temp).
        dt[k] is the time from sample k to sample k+1.
        Returns (v_terminal, states) with states of shape (N, n_states).
        """
        current = np.asarray(current, dtype=float)
        n = current.size
        dt = np.broadcast_to(np.asarray(dt, dtype=float), (n,))
        temp = np.broadcast_to(np.asarray(temp, dtype=float), (n,))
        x = np.zeros(self.n_states)
        x[0] = soc0
        if v_rc0 is not None:
            x[1:] = v_rc0
        v = np.empty(n)
        states = np.empty((n, self.n_states))
        for k in range(n):
            states[k] = x
            v[k], _ = self.output(x, current[k], temp[k])
            x, _ = self.transition(x, current[k], dt[k], temp[k])
        return v, states
