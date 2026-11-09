"""Calibrate the slow polarisation branch on the TRAINING drive cycles.

Why: the slow branch fitted on the HPPC relaxations is poorly identified (time
constants scatter from 100 s to 10 000 s between SoC levels) and overshoots on
drive cycles (results/tables/m2_ecm_selection.md, variants 3rc and 3rc_bv).
The neural networks learn from the training drive cycles, so the physics model
may use the same cycles. Validation, test and all 10 degC data stay untouched.

What is fitted (everything else stays as identified from HPPC and C/20):
  for each of the five model temperatures j:
      c_j        constant OCV offset relative to the base OCV curve (V); two base
                 shapes are tried, the C/20 average and the C/20 discharge + IR curve
      Rs_j(z)    slow-branch resistance, linear in SoC: a_j * z + b_j * (1 - z), a_j, b_j >= 0
  one slow time constant tau for all temperatures (grid search)
  optionally one thermal gain g (grid search): parameters are looked up at
      T_eff = T + g * (T - T_start)
  where T_start is the rested temperature at the start of the cycle. For a
  cell with internal and external thermal resistance in series this is the
  steady-state core temperature, with g = R_internal / R_external.

For fixed tau and g the model is linear in the 15 unknowns, because the model
interpolates OCV and resistances linearly in temperature (hat functions h_j):
  v_meas - v_fast = sum_j h_j(T) c_j - sum_j [ a_j RC(h_j z i) + b_j RC(h_j (1 - z) i) ]
with RC(.) a first-order filter of time constant tau. Solved by bounded linear
least squares (scipy.optimize.lsq_linear).

Adds four variants (with or without Butler-Volmer, two base OCV shapes) to
results/characterization/variants.json. Writes results/tables/m2_slow_calibration.md.
Run after m2_characterize.py: python experiments/m2_calibrate_slow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socest.cell import CHAR, capacity_fn, load_ocv  # noqa: E402
from socest.cycles import load_split  # noqa: E402
from socest.datasets.lg_hg2 import FOLDER_TEMPS  # noqa: E402
from socest.ecm import TheveninModel  # noqa: E402
from socest.params import ParamTable  # noqa: E402
from socest.split import HELD_OUT_FOLDER  # noqa: E402
from socest.thermal import effective_temperature  # noqa: E402

TAU_GRID = np.geomspace(100.0, 5000.0, 18)
G_GRID = (0.0, 0.5, 1.0)
FOLDERS = [f for f in FOLDER_TEMPS if f != HELD_OUT_FOLDER]
NODES = np.array([FOLDER_TEMPS[f] for f in FOLDERS])


def hats(T):
    """Linear-interpolation weights of each temperature node, clamped at the ends. Shape (N, m)."""
    t = np.clip(T, NODES[0], NODES[-1])
    H = np.zeros((t.size, NODES.size))
    j = np.clip(np.searchsorted(NODES, t, side="right") - 1, 0, NODES.size - 2)
    w = (t - NODES[j]) / (NODES[j + 1] - NODES[j])
    H[np.arange(t.size), j] = 1 - w
    H[np.arange(t.size), j + 1] = w
    return H


def rc(x, tau):
    """State of a unit-resistance RC branch before each sample (matches TheveninModel)."""
    a = np.exp(-1.0 / tau)
    y = lfilter([1 - a], [1, -a], x, axis=0)
    return np.concatenate([np.zeros((1,) + x.shape[1:]), y[:-1]], axis=0)


def calibrate(fast_params: str, base_ocv: str, data, cap_of):
    ocv = load_ocv(base_ocv)
    table = ParamTable.from_json(CHAR / fast_params)
    best = None
    log = []
    for g in G_GRID:
        prep = []
        for r, u in data:
            i, v, T = u.current_a.to_numpy(), u.voltage_v.to_numpy(), u.temp_degc.to_numpy()
            Te = effective_temperature(T, g)
            v_fast, st = TheveninModel(ocv, table, cap_of(T[0])).simulate(i, 1.0, Te, soc0=1.0)
            prep.append((v - v_fast, hats(Te), st[:, 0], i))
        y = np.concatenate([p[0] for p in prep])
        for tau in TAU_GRID:
            cols = []
            for _, H, z, i in prep:
                cols.append(np.hstack([H, -rc(H * (z * i)[:, None], tau), -rc(H * ((1 - z) * i)[:, None], tau)]))
            A = np.vstack(cols)
            m = NODES.size
            lb = np.r_[np.full(m, -np.inf), np.zeros(2 * m)]
            sol = lsq_linear(A, y, bounds=(lb, np.full(3 * m, np.inf)))
            rmse = float(np.sqrt(np.mean((A @ sol.x - y) ** 2)))
            log.append({"g": g, "tau_s": tau, "train_rmse_mv": 1000 * rmse})
            if best is None or rmse < best["rmse"]:
                best = {"rmse": rmse, "g": g, "tau": float(tau), "c": sol.x[:m], "a": sol.x[m:2 * m], "b": sol.x[2 * m:]}
        print(f"  {fast_params} g={g}: best train RMSE so far {1000 * best['rmse']:.1f} mV", flush=True)
    return best, pd.DataFrame(log), table


def write_variant(name, description, fast_table: ParamTable, fit, base_ocv):
    socs = fast_table.socs
    r_slow = fit["a"][:, None] * socs[None, :] + fit["b"][:, None] * (1 - socs[None, :])
    r_slow = np.maximum(r_slow, 1e-6)
    r = np.concatenate([fast_table.r, r_slow[None]], axis=0)
    tau = np.concatenate([fast_table.tau, np.full((1,) + r_slow.shape, fit["tau"])], axis=0)
    i0 = fast_table.i0 if fast_table._log_i0 is not None else None
    ParamTable(fast_table.temps, socs, fast_table.r0, r, tau, i0, fast_table.bv_n).to_json(CHAR / f"params_{name}.json")

    base = json.loads((CHAR / base_ocv).read_text())
    all_t = np.array(base["temps"])
    c_all = np.interp(all_t, NODES, fit["c"])     # the 10 degC row is never loaded by the model
    base["ocv"] = (np.array(base["ocv"]) + c_all[:, None]).tolist()
    base["method"] = base["method"] + "; plus offset calibrated on training drive cycles"
    (CHAR / f"ocv_{name}.json").write_text(json.dumps(base))

    v = json.loads((CHAR / "variants.json").read_text())
    v[name] = {"params": f"params_{name}.json", "ocv": f"ocv_{name}.json", "description": description,
               "thermal_gain": fit["g"], "calibrated_on": "training drive cycles (slow branch, OCV offset)"}
    (CHAR / "variants.json").write_text(json.dumps(v, indent=1))


def main():
    cap_of = capacity_fn()
    data = load_split("train")
    md = ["# Slow-branch calibration on the training drive cycles", "", __doc__.split("What is fitted")[0].strip().split("\n\n", 1)[1], ""]
    runs = [
        ("3rc_dc_avg", "params_2rc.json", "ocv_average.json", "2RC (HPPC) + calibrated slow branch, OCV shape: C/20 average"),
        ("3rc_dc_dis", "params_2rc.json", "ocv_disir.json", "2RC (HPPC) + calibrated slow branch, OCV shape: C/20 discharge + IR"),
        ("3rc_bv_dc_avg", "params_2rc_bv.json", "ocv_average.json",
         "2RC + Butler-Volmer (HPPC) + calibrated slow branch, OCV shape: C/20 average"),
        ("3rc_bv_dc_dis", "params_2rc_bv.json", "ocv_disir.json",
         "2RC + Butler-Volmer (HPPC) + calibrated slow branch, OCV shape: C/20 discharge + IR"),
    ]
    for name, fast, base_ocv, desc in runs:
        fit, log, table = calibrate(fast, base_ocv, data, cap_of)
        write_variant(name, desc, table, fit, base_ocv)
        tab = pd.DataFrame({"OCV offset c (mV)": 1000 * fit["c"], "R slow at SoC 1 (mΩ)": 1000 * fit["a"],
                            "R slow at SoC 0 (mΩ)": 1000 * fit["b"]}, index=[f"{t:g} °C" for t in NODES])
        g_tab = log.loc[log.groupby("g").train_rmse_mv.idxmin()].set_index("g")
        md += [f"## `{name}`: {desc}", "",
               f"Chosen: tau = {fit['tau']:.0f} s, thermal gain g = {fit['g']}, training voltage RMSE {1000 * fit['rmse']:.1f} mV.", "",
               tab.round(1).to_markdown(), "", "Best training RMSE for each thermal gain:", "", g_tab.round(1).to_markdown(), ""]
        print(name, f"tau {fit['tau']:.0f} s, g {fit['g']}, train RMSE {1000 * fit['rmse']:.1f} mV")
        print(tab.round(1).to_string())
        print(g_tab.round(1).to_string())
    (ROOT / "results" / "tables" / "m2_slow_calibration.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
