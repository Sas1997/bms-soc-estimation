"""Open-circuit voltage (OCV) as a function of SoC and, optionally, temperature.

The curve is stored as a dense lookup table so evaluation is a single np.interp
call. This keeps the per-step cost low enough for embedded targets.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


class OCVCurve:
    """OCV(soc, temp) built from one SoC-OCV curve per temperature.

    Parameters
    ----------
    soc : array of shape (n,)
        SoC breakpoints in [0, 1], strictly increasing.
    ocv : array of shape (n,) or (m, n)
        OCV in volts. One row per temperature.
    temps : array of shape (m,), optional
        Temperatures in degC for each row of ``ocv``. Omit for a single curve.
    n_dense : int
        Resolution of the dense lookup table.

    Between temperatures the OCV is linearly interpolated. Outside the SoC or
    temperature range the value is held at the nearest edge.
    """

    def __init__(self, soc, ocv, temps=None, n_dense: int = 2001):
        soc = np.asarray(soc, dtype=float)
        ocv = np.atleast_2d(np.asarray(ocv, dtype=float))
        if np.any(np.diff(soc) <= 0):
            raise ValueError("soc breakpoints must be strictly increasing")
        if ocv.shape[1] != soc.size:
            raise ValueError("ocv rows must match the length of soc")
        if temps is None:
            if ocv.shape[0] != 1:
                raise ValueError("temps is required when ocv has more than one row")
            temps = np.array([25.0])
        temps = np.asarray(temps, dtype=float)
        order = np.argsort(temps)
        self.temps = temps[order]
        ocv = ocv[order]

        self.grid = np.linspace(0.0, 1.0, n_dense)
        self.table = np.empty((self.temps.size, n_dense))
        self.dtable = np.empty_like(self.table)
        for k, row in enumerate(ocv):
            f = PchipInterpolator(soc, row, extrapolate=False)
            g = np.clip(self.grid, soc[0], soc[-1])
            self.table[k] = f(g)
            self.dtable[k] = np.gradient(self.table[k], self.grid)

    def _row_weights(self, temp):
        if self.temps.size == 1 or temp is None:
            return 0, 0, 0.0
        t = float(np.clip(temp, self.temps[0], self.temps[-1]))
        j = int(np.searchsorted(self.temps, t, side="right")) - 1
        j = min(max(j, 0), self.temps.size - 2)
        w = (t - self.temps[j]) / (self.temps[j + 1] - self.temps[j])
        return j, j + 1, w

    def _eval(self, table, soc, temp):
        j0, j1, w = self._row_weights(temp)
        a = np.interp(soc, self.grid, table[j0])
        if w == 0.0:
            return a
        b = np.interp(soc, self.grid, table[j1])
        return (1.0 - w) * a + w * b

    def __call__(self, soc, temp=None):
        """OCV in volts."""
        return self._eval(self.table, soc, temp)

    def dsoc(self, soc, temp=None):
        """dOCV/dSoC in volts per unit SoC."""
        return self._eval(self.dtable, soc, temp)
