"""ECM parameters tabulated over (temperature, SoC) with bilinear interpolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ecm import ECMParams


def _bilinear(table: np.ndarray, temps: np.ndarray, socs: np.ndarray, temp: float, soc: float) -> float:
    """Bilinear interpolation on a (len(temps), len(socs)) table, clamped at the edges."""
    t = float(np.clip(temp, temps[0], temps[-1]))
    s = float(np.clip(soc, socs[0], socs[-1]))
    if temps.size == 1:
        return float(np.interp(s, socs, table[0]))
    j = min(max(int(np.searchsorted(temps, t, side="right")) - 1, 0), temps.size - 2)
    w = (t - temps[j]) / (temps[j + 1] - temps[j])
    a = np.interp(s, socs, table[j])
    b = np.interp(s, socs, table[j + 1])
    return float((1 - w) * a + w * b)


class ParamTable:
    """Callable (soc, temp) -> ECMParams.

    temps : (m,) degC, ascending
    socs  : (n,) SoC breakpoints, ascending
    r0    : (m, n) ohm
    r     : (k, m, n) ohm, one slice per RC branch
    tau   : (k, m, n) s
    Time constants are interpolated in log space, resistances linearly.
    """

    def __init__(self, temps, socs, r0, r, tau):
        self.temps = np.asarray(temps, dtype=float)
        self.socs = np.asarray(socs, dtype=float)
        self.r0 = np.asarray(r0, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.tau = np.asarray(tau, dtype=float)
        m, n = self.temps.size, self.socs.size
        if self.r0.shape != (m, n) or self.r.shape[1:] != (m, n) or self.tau.shape != self.r.shape:
            raise ValueError("table shapes do not match temps and socs")
        self._log_tau = np.log(self.tau)

    @property
    def n_rc(self) -> int:
        return self.r.shape[0]

    def __call__(self, soc: float, temp: float) -> ECMParams:
        T, S = self.temps, self.socs
        r0 = _bilinear(self.r0, T, S, temp, soc)
        r = tuple(_bilinear(self.r[i], T, S, temp, soc) for i in range(self.n_rc))
        tau = tuple(float(np.exp(_bilinear(self._log_tau[i], T, S, temp, soc))) for i in range(self.n_rc))
        return ECMParams(r0=r0, r=r, tau=tau)

    def to_json(self, path):
        data = {k: getattr(self, k).tolist() for k in ("temps", "socs", "r0", "r", "tau")}
        Path(path).write_text(json.dumps(data, indent=1))

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["temps"], d["socs"], d["r0"], d["r"], d["tau"])
