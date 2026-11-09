"""ECM parameters tabulated over (temperature, SoC) with bilinear interpolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ecm import ECMParams


class ParamTable:
    """Callable (soc, temp) -> ECMParams.

    temps : (m,) degC, ascending
    socs  : (n,) SoC breakpoints, ascending
    r0    : (m, n) ohm
    r     : (k, m, n) ohm, one slice per RC branch
    tau   : (k, m, n) s
    i0    : (m,) A, optional Butler-Volmer exchange current per temperature
            (inf or None = linear model)
    bv_n  : series interfaces in the Butler-Volmer term (see socest.ecm)
    Resistances are interpolated linearly; time constants and i0 in log space.
    Values are held at the edges outside the tabulated range.
    """

    def __init__(self, temps, socs, r0, r, tau, i0=None, bv_n=1):
        self.temps = np.asarray(temps, dtype=float)
        self.socs = np.asarray(socs, dtype=float)
        self.r0 = np.asarray(r0, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.tau = np.asarray(tau, dtype=float)
        m, n = self.temps.size, self.socs.size
        if self.r0.shape != (m, n) or self.r.shape[1:] != (m, n) or self.tau.shape != self.r.shape:
            raise ValueError("table shapes do not match temps and socs")
        self.i0 = None if i0 is None else np.asarray(i0, dtype=float)
        self.bv_n = int(bv_n)
        if self.i0 is not None and self.i0.shape != (m,):
            raise ValueError("i0 must have one value per temperature")
        k = self.r.shape[0]
        # One stacked array so a lookup is a single bilinear blend: [r0, r_1..r_k, log tau_1..k]
        self._stack = np.concatenate([self.r0[None], self.r, np.log(self.tau)], axis=0)
        self._k = k
        if self.i0 is not None and np.all(np.isfinite(self.i0)):
            self._log_i0 = np.log(self.i0)
        else:
            self._log_i0 = None

    @property
    def n_rc(self) -> int:
        return self._k

    def __call__(self, soc: float, temp: float) -> ECMParams:
        T, S = self.temps, self.socs
        t = min(max(temp, T[0]), T[-1])
        s = min(max(soc, S[0]), S[-1])
        if T.size == 1:
            j, w = 0, 0.0
        else:
            j = min(max(int(np.searchsorted(T, t, side="right")) - 1, 0), T.size - 2)
            w = (t - T[j]) / (T[j + 1] - T[j])
        q = min(max(int(np.searchsorted(S, s, side="right")) - 1, 0), S.size - 2)
        u = (s - S[q]) / (S[q + 1] - S[q])
        A = self._stack
        lo = (1 - u) * A[:, j, q] + u * A[:, j, q + 1]
        if w:
            hi = (1 - u) * A[:, j + 1, q] + u * A[:, j + 1, q + 1]
            val = (1 - w) * lo + w * hi
        else:
            val = lo
        k = self._k
        i0 = float("inf")
        if self._log_i0 is not None:
            i0 = float(np.exp((1 - w) * self._log_i0[j] + w * self._log_i0[min(j + 1, T.size - 1)]))
        return ECMParams(r0=float(val[0]), r=tuple(val[1 : 1 + k].tolist()),
                         tau=tuple(np.exp(val[1 + k :]).tolist()), i0=i0, bv_n=self.bv_n)

    def to_json(self, path):
        data = {key: getattr(self, key).tolist() for key in ("temps", "socs", "r0", "r", "tau")}
        data["i0"] = None if self._log_i0 is None else self.i0.tolist()
        data["bv_n"] = self.bv_n
        Path(path).write_text(json.dumps(data, indent=1))

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["temps"], d["socs"], d["r0"], d["r"], d["tau"], d.get("i0"), d.get("bv_n", 1))
