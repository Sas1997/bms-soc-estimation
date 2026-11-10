"""Series pack of identified Thevenin cells with cell-balancing circuits.

Pack model
----------
N cells in series, each one the identified cell model (socest.cell) at a fixed
temperature, with its own capacity and resistance scale factors to represent
cell-to-cell variation. Arrays have shape (P, N): P independent packs simulated
at once (for Monte Carlo runs), N cells per pack. The equations and the
discretisation are those of socest.ecm.TheveninModel (checked in
tests/test_pack.py); OCV and parameters are tabulated on a fine SoC grid at the
pack temperature.

Balancing circuits (averaged models)
------------------------------------
Each circuit returns the extra current it draws from every cell (positive =
taken from the cell) and the power it dissipates. Switching ripple, converter
standby consumption and current-dependent efficiency are not modelled.

  Passive      one shunt resistor per cell, switched on for cells above the
               lowest SoC (dissipative equalisation).
  Adjacent     one bidirectional DC-DC converter between each pair of
               neighbouring cells, moving charge from the higher-SoC cell to the
               lower one (cell-to-cell, N - 1 converters).
  Cell-to-pack one converter per cell; a cell above the pack mean SoC feeds the
               whole string (cell-to-pack).

Each circuit acts on cells whose SoC differs from its reference (pack minimum,
neighbour, or pack mean) by more than a dead band of 0.01 points. The dead band
must stay well below the balancing target divided by the string length, or the
mean-referenced and neighbour-referenced controllers stall before the target.

Topology classes follow M. Daowd, N. Omar, P. Van Den Bossche, J. Van Mierlo,
"Passive and active battery balancing comparison based on MATLAB simulation",
IEEE VPPC 2011, and J. Gallardo-Lozano et al., "Battery equalization active
methods", J. Power Sources 246 (2014) 934-949. Converter efficiency and
balancing current are inputs, not values taken from those papers.

Current convention: positive = discharge, as everywhere in this package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PackModel:
    def __init__(self, cell_model, temp_c: float, capacity_scale, resistance_scale, n_grid: int = 2001):
        """cell_model: socest.ecm.TheveninModel (its capacity is the nominal cell capacity).
        capacity_scale, resistance_scale: arrays (P, N) multiplying capacity and every resistance."""
        g = np.linspace(0.0, 1.0, n_grid)
        self.grid = g
        self.temp_c = float(temp_c)
        self.ocv_t = np.asarray(cell_model.ocv(g, temp_c), dtype=float)
        ps = [cell_model.params(float(s), temp_c) for s in g]
        if any(np.isfinite(p.i0) for p in ps):
            raise NotImplementedError("PackModel does not include the Butler-Volmer term")
        self.r0_t = np.array([p.r0 for p in ps])
        self.r_t = np.array([p.r for p in ps]).T          # (K, n_grid)
        self.tau_t = np.array([p.tau for p in ps]).T
        self.n_rc = self.r_t.shape[0]
        self.cap = cell_model.capacity_ah * np.asarray(capacity_scale, dtype=float)
        self.k_r = np.asarray(resistance_scale, dtype=float)
        if self.cap.shape != self.k_r.shape or self.cap.ndim != 2:
            raise ValueError("capacity_scale and resistance_scale must both have shape (P, N)")
        # Stored energy per Ah of capacity from SoC 0 to s (V * fraction): integral of OCV over SoC.
        self.e_t = np.concatenate([[0.0], np.cumsum(0.5 * (self.ocv_t[1:] + self.ocv_t[:-1]) * np.diff(g))])
        self.soc = None
        self.vrc = None

    @property
    def shape(self):
        return self.cap.shape

    def reset(self, soc0):
        soc0 = np.asarray(soc0, dtype=float)
        if soc0.shape != self.shape:
            raise ValueError(f"soc0 must have shape {self.shape}")
        self.soc = soc0.copy()
        self.vrc = np.zeros(self.shape + (self.n_rc,))

    def _lut(self, table):
        return np.interp(self.soc.ravel(), self.grid, table).reshape(self.shape)

    def voltage(self, current):
        """Terminal voltage of every cell for cell currents (P, N), before the step."""
        return self._lut(self.ocv_t) - self.vrc.sum(-1) - self.k_r * self._lut(self.r0_t) * current

    def step(self, current, dt: float):
        """Apply cell currents (P, N) for dt seconds. Returns the terminal voltages during the step."""
        v = self.voltage(current)
        for k in range(self.n_rc):
            a = np.exp(-dt / self._lut(self.tau_t[k]))
            self.vrc[..., k] = a * self.vrc[..., k] + self.k_r * self._lut(self.r_t[k]) * (1.0 - a) * current
        self.soc = self.soc - current * dt / (3600.0 * self.cap)
        return v

    def stored_energy_wh(self):
        """Chemical energy stored in each cell above SoC 0 (Wh), from the OCV curve. Shape (P, N)."""
        return self.cap * self._lut(self.e_t)


# ---------------------------------------------------------------- balancing circuits
@dataclass
class Passive:
    r_bleed: float = 20.0          # ohm; about 0.2 A at 4 V
    name: str = "Passive (shunt resistor)"

    def currents(self, soc_ctrl, v, enabled, deadband):
        bleed = enabled[:, None] & (soc_ctrl - soc_ctrl.min(axis=1, keepdims=True) > deadband)
        i = np.where(bleed, v / self.r_bleed, 0.0)
        return i, (i * v).sum(axis=1)


@dataclass
class Adjacent:
    i_bal: float = 0.2             # A drawn from the donor cell
    eta: float = 0.85              # converter efficiency (power out / power in)
    name: str = "Active, adjacent cell-to-cell"

    def currents(self, soc_ctrl, v, enabled, deadband):
        P, N = soc_ctrl.shape
        i = np.zeros((P, N))
        loss = np.zeros(P)
        for j in range(N - 1):
            d = soc_ctrl[:, j] - soc_ctrl[:, j + 1]
            for mask, a, b in ((enabled & (d > deadband), j, j + 1), (enabled & (d < -deadband), j + 1, j)):
                p_in = np.where(mask, v[:, a] * self.i_bal, 0.0)
                i[:, a] += np.where(mask, self.i_bal, 0.0)
                i[:, b] -= self.eta * p_in / v[:, b]
                loss += (1.0 - self.eta) * p_in
        return i, loss


@dataclass
class CellToPack:
    i_bal: float = 0.2
    eta: float = 0.85
    name: str = "Active, cell-to-pack"

    def currents(self, soc_ctrl, v, enabled, deadband):
        donor = enabled[:, None] & (soc_ctrl - soc_ctrl.mean(axis=1, keepdims=True) > deadband)
        p_in = np.where(donor, v * self.i_bal, 0.0)
        i_back = self.eta * p_in.sum(axis=1) / v.sum(axis=1)       # charging current into the whole string
        i = np.where(donor, self.i_bal, 0.0) - i_back[:, None]
        return i, (1.0 - self.eta) * p_in.sum(axis=1)


# ---------------------------------------------------------------- scenarios
def simulate_rest(pack: PackModel, balancer, soc0, ctrl_offset=0.0, dt=10.0, t_max_h=48.0,
                  stop=0.005, deadband=0.0001, record=None):
    """Balance a resting pack until the controller's SoC spread falls below `stop`.

    The controller sees SoC + ctrl_offset (a fixed per-cell estimation error).
    Returns dict of arrays (P,): t_balance_h (nan if not reached), loss_wh (drop in
    stored energy), circuit_loss_wh, final_spread (true SoC, fraction); plus
    'trace' (time_h, soc) of pack `record` if given.
    """
    pack.reset(soc0)
    P = pack.shape[0]
    e0 = pack.stored_energy_wh().sum(axis=1)
    zero = np.zeros(pack.shape)
    v = pack.voltage(zero)
    t_done = np.full(P, np.nan)
    done = np.zeros(P, bool)
    circuit = np.zeros(P)
    tr_t, tr_s = [], []
    for k in range(int(t_max_h * 3600 / dt) + 1):
        if record is not None:
            tr_t.append(k * dt / 3600.0)
            tr_s.append(pack.soc[record].copy())
        ctrl = pack.soc + ctrl_offset
        newly = ~done & (ctrl.max(axis=1) - ctrl.min(axis=1) < stop)
        t_done[newly] = k * dt / 3600.0
        done |= newly
        if done.all():
            break
        i, loss = balancer.currents(ctrl, v, ~done, deadband)
        circuit += loss * dt / 3600.0
        v = pack.step(i, dt)
    out = {"t_balance_h": t_done, "loss_wh": e0 - pack.stored_energy_wh().sum(axis=1),
           "circuit_loss_wh": circuit, "final_spread": pack.soc.max(axis=1) - pack.soc.min(axis=1),
           "stored_wh_start": e0}
    if record is not None:
        out["trace"] = (np.array(tr_t), np.array(tr_s))
    return out


def simulate_drive(pack: PackModel, balancer, soc0, i_profile, dt=1.0, v_cut=2.8, ctrl_offset=0.0,
                   start=0.01, stop=0.005, deadband=0.0001, max_repeats=5, record=None):
    """Discharge the pack with a repeating current profile until the first cell falls below v_cut.

    balancer None = no balancing. Balancing runs during the discharge with
    hysteresis on the controller's SoC spread (on above `start`, off below `stop`).
    Returns dict of arrays (P,): energy_wh and ah delivered to the load, circuit_loss_wh,
    t_end_h, soc_end (P, N); plus 'trace' (time_h, soc) of pack `record` if given.
    """
    pack.reset(soc0)
    P = pack.shape[0]
    i_profile = np.asarray(i_profile, dtype=float)
    zero = np.zeros(pack.shape)
    v = pack.voltage(zero)
    done = np.zeros(P, bool)
    enabled = np.zeros(P, bool)
    e_del, ah_del, circuit = np.zeros(P), np.zeros(P), np.zeros(P)
    t_end = np.full(P, np.nan)
    tr_t, tr_s = [], []
    for k in range(max_repeats * len(i_profile)):
        ip = i_profile[k % len(i_profile)]
        ctrl = pack.soc + ctrl_offset
        spread = ctrl.max(axis=1) - ctrl.min(axis=1)
        enabled = np.where(enabled, spread > stop, spread > start) & ~done
        if balancer is None:
            ib, loss = zero, np.zeros(P)
        else:
            ib, loss = balancer.currents(ctrl, v, enabled, deadband)
        i_cell = np.where(done[:, None], 0.0, ip + ib)
        v = pack.step(i_cell, dt)
        live = ~done
        e_del += np.where(live, v.sum(axis=1) * ip * dt / 3600.0, 0.0)
        ah_del += np.where(live, ip * dt / 3600.0, 0.0)
        circuit += np.where(live, loss * dt / 3600.0, 0.0)
        hit = live & (v.min(axis=1) < v_cut)
        t_end[hit] = (k + 1) * dt / 3600.0
        done |= hit
        if record is not None and k % 30 == 0:
            tr_t.append(k * dt / 3600.0)
            tr_s.append(pack.soc[record].copy())
        if done.all():
            break
    out = {"energy_wh": e_del, "ah": ah_del, "circuit_loss_wh": circuit, "t_end_h": t_end, "soc_end": pack.soc.copy()}
    if record is not None:
        out["trace"] = (np.array(tr_t), np.array(tr_s))
    return out
