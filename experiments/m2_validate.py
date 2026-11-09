"""M2 validation on drive cycles (all at 1 Hz, see socest.resample).

1. ECM voltage prediction: open-loop simulation of terminal voltage from current
   and measured temperature, starting from SoC = 1, for 1RC and 2RC models.
   Voltage RMSE per cycle. HPPC data were used for fitting; drive cycles never were.
2. Coulomb counting baseline on both test sets: correct initial SoC, initial SoC
   off by -20 points, and a +50 mA current sensor bias.

Writes results/tables/m2_*.md, results/m2_*.csv and results/figures/m2_*.png.
Run: python experiments/m2_validate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socest.cell import PROC, capacity_fn, load_model  # noqa: E402
from socest.coulomb import coulomb_count  # noqa: E402
from socest.metrics import soc_errors  # noqa: E402
from socest.resample import to_uniform  # noqa: E402
from socest.split import load_index  # noqa: E402

RES, TAB, FIG = ROOT / "results", ROOT / "results" / "tables", ROOT / "results" / "figures"
ORDER = ["n20degC", "n10degC", "0degC", "10degC", "25degC", "40degC"]


def cycles():
    ix = load_index(PROC)
    for _, r in ix.iterrows():
        yield r, to_uniform(pd.read_parquet(PROC / r.parquet), 1.0)


def main():
    cap_of = capacity_fn()
    models = {}
    ecm_rows, cc_rows, keep = [], [], {}
    for r, u in cycles():
        i, v, T = u.current_a.to_numpy(), u.voltage_v.to_numpy(), u.temp_degc.to_numpy()
        cap = cap_of(T[0])
        for n_rc in (1, 2):
            m = load_model(n_rc, capacity_ah=cap)
            v_sim, _ = m.simulate(i, 1.0, T, soc0=1.0)
            err = v_sim - v
            ecm_rows.append({"folder": r.folder, "test": r.test, "split": r.split, "n_rc": n_rc,
                             "v_rmse_mv": 1000 * np.sqrt(np.mean(err**2)),
                             "v_rmse_soc_gt_10pct_mv": 1000 * np.sqrt(np.mean(err[u.soc_ref > 0.1] ** 2))})
            if r.folder in ("n20degC", "25degC", "10degC") and r.test == "UDDS" and n_rc == 2:
                keep[r.folder] = (u, v_sim)
        if r.split.startswith("test"):
            for case, soc0, bias in (("correct init", 1.0, 0.0), ("init -20 pts", 0.8, 0.0), ("+50 mA bias", 1.0, 0.05)):
                soc = coulomb_count(i + bias, 1.0, soc0, cap)
                e = soc_errors(soc, u.soc_ref)
                cc_rows.append({"folder": r.folder, "test": r.test, "split": r.split, "case": case,
                                **{k: 100 * val for k, val in e.items()}})
        print(f"  {r.folder}/{r.test} done", flush=True)

    ecm = pd.DataFrame(ecm_rows)
    cc = pd.DataFrame(cc_rows)
    ecm.to_csv(RES / "m2_ecm_drive_cycles.csv", index=False)
    cc.to_csv(RES / "m2_coulomb_counting.csv", index=False)

    ecm["folder"] = pd.Categorical(ecm.folder, ORDER, ordered=True)
    t1 = ecm.pivot_table(index="folder", columns="n_rc", values="v_rmse_mv", aggfunc="median", observed=True)
    t1.columns = [f"{c}RC median V RMSE (mV)" for c in t1.columns]
    t1b = ecm.pivot_table(index="folder", columns="n_rc", values="v_rmse_soc_gt_10pct_mv", aggfunc="median", observed=True)
    t1b.columns = [f"{c}RC, SoC > 10 % (mV)" for c in t1b.columns]
    t1 = t1.join(t1b)
    t2 = cc.groupby(["split", "case"], sort=False)[["rmse", "mae", "max_abs"]].mean()
    md = ["# ECM open-loop voltage prediction on drive cycles (1 Hz)", "",
          "Parameters from HPPC; OCV from C/20; 10 degC values interpolated, never fitted. "
          "Median over cycles at each temperature.", "", t1.round(1).to_markdown(), "",
          "# Coulomb counting on the test sets (SoC error, percentage points, mean over cycles)", "",
          "The reference SoC is itself built from the tester Ah counter with a known full-charge start, so "
          "coulomb counting with the correct initial SoC is near-exact by construction; it is shown to make "
          "that explicit. Its weaknesses are an unknown initial SoC and sensor bias, which it never corrects.", "",
          t2.round(2).to_markdown(), ""]
    (TAB / "m2_validation.md").write_text("\n".join(md))
    print(t1.round(1).to_string())
    print(t2.round(2).to_string())

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    for a, f in zip(ax, ["n20degC", "10degC", "25degC"]):
        u, vs = keep[f]
        sl = slice(3600, 3600 + 600)
        a.plot(u.time_s[sl], u.voltage_v[sl], "k", lw=1, label="measured")
        a.plot(u.time_s[sl], vs[sl], "tab:blue", lw=1, label="2RC model (open loop)")
        rm = ecm[(ecm.folder == f) & (ecm.test == "UDDS") & (ecm.n_rc == 2)].v_rmse_mv.iloc[0]
        a.set(title=f"UDDS at {f.replace('n', '-').replace('degC', ' °C')}: 10 min window after 1 h, full-cycle RMSE {rm:.0f} mV"
              + (" (held-out temperature)" if f == "10degC" else ""), ylabel="V")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(FIG / "m2_ecm_drive_cycle.png", dpi=110)


if __name__ == "__main__":
    main()
