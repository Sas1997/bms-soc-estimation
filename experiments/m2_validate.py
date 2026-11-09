"""M2 validation on every usable drive cycle (1 Hz), reported per split.

1. ECM voltage prediction: open-loop simulation of terminal voltage from current
   and temperature, starting from SoC = 1, for the 1RC and 2RC baselines and the
   selected model (results/characterization/selected.json).
2. Coulomb counting baseline on both test sets: correct initial SoC, initial SoC
   off by -20 points, and a +50 mA current sensor bias.

Nothing is fitted or selected here.
Writes results/tables/m2_validation.md, results/m2_*.csv and results/figures/m2_ecm_drive_cycle.png.
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

from socest.cell import capacity_fn, load_model, selected_variant, variants  # noqa: E402
from socest.coulomb import coulomb_count  # noqa: E402
from socest.cycles import load_split  # noqa: E402
from socest.metrics import soc_errors  # noqa: E402
from socest.split import SPLITS  # noqa: E402
from socest.thermal import effective_temperature  # noqa: E402

RES, TAB, FIG = ROOT / "results", ROOT / "results" / "tables", ROOT / "results" / "figures"
ORDER = ["n20degC", "n10degC", "0degC", "10degC", "25degC", "40degC"]
NICE = {"n20degC": "-20 °C", "n10degC": "-10 °C", "0degC": "0 °C", "10degC": "10 °C (held out)", "25degC": "25 °C", "40degC": "40 °C"}


def main():
    cap_of = capacity_fn()
    sel = selected_variant()
    names = ["1rc", "2rc", sel]
    ecm_rows, cc_rows, keep = [], [], {}
    for r, u in load_split(*SPLITS):
        i, v, T = u.current_a.to_numpy(), u.voltage_v.to_numpy(), u.temp_degc.to_numpy()
        cap = cap_of(T[0])
        for name in names:
            m = load_model(name, capacity_ah=cap)
            v_sim, _ = m.simulate(i, 1.0, effective_temperature(T, m.thermal_gain), soc0=1.0)
            err = v_sim - v
            ecm_rows.append({"folder": r.folder, "test": r.test, "split": r.split, "model": name,
                             "v_rmse_mv": 1000 * np.sqrt(np.mean(err**2)), "bias_mv": 1000 * err.mean()})
            if r.test == "UDDS" and r.folder in ("n20degC", "10degC", "25degC"):
                keep[(r.folder, name)] = (u, v_sim)
        if r.split.startswith("test"):
            for case, soc0, bias in (("correct init", 1.0, 0.0), ("init -20 pts", 0.8, 0.0), ("+50 mA bias", 1.0, 0.05)):
                e = soc_errors(coulomb_count(i + bias, 1.0, soc0, cap), u.soc_ref)
                cc_rows.append({"folder": r.folder, "test": r.test, "split": r.split, "case": case,
                                **{k: 100 * val for k, val in e.items()}})
    ecm, cc = pd.DataFrame(ecm_rows), pd.DataFrame(cc_rows)
    ecm.to_csv(RES / "m2_ecm_drive_cycles.csv", index=False)
    cc.to_csv(RES / "m2_coulomb_counting.csv", index=False)

    ecm["temperature"] = pd.Categorical(ecm.folder.map(NICE), [NICE[f] for f in ORDER], ordered=True)
    desc = {n: variants()[n]["description"] for n in names}

    def pivot(sub):
        t = sub.pivot_table(index="temperature", columns="model", values="v_rmse_mv", aggfunc="median", observed=True)
        return t[names]

    test = ecm[ecm.split.str.startswith("test")]
    t_test, t_all = pivot(test), pivot(ecm)
    t2 = cc.groupby(["split", "case"], sort=False)[["rmse", "mae", "max_abs"]].mean()
    md = ["# ECM open-loop voltage prediction on drive cycles (1 Hz)", "",
          "Median voltage RMSE over cycles (mV). Models:", ""]
    md += [f"- `{n}`: {d}" for n, d in desc.items()]
    md += ["", "At 10 °C every model interpolates its parameters between 0 and 25 °C; no 10 °C data was used anywhere.", "",
           "## Test cycles only (UDDS, HWFET, LA92, US06, and all cycles at 10 °C)", "", t_test.round(1).to_markdown(), "",
           "## All usable cycles (train, validation and test)", "", t_all.round(1).to_markdown(), "",
           "# Coulomb counting on the test sets (SoC error, percentage points, mean over cycles)", "",
           "The reference SoC is itself built from the tester Ah counter with a known full-charge start, so "
           "coulomb counting with the correct initial SoC is near-exact by construction; it is shown to make "
           "that explicit. Its weaknesses are an unknown initial SoC and sensor bias, which it never corrects.", "",
           t2.round(2).to_markdown(), ""]
    (TAB / "m2_validation.md").write_text("\n".join(md))
    print("TEST\n" + t_test.round(1).to_string())
    print("ALL\n" + t_all.round(1).to_string())
    print(t2.round(2).to_string())

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    for a, f in zip(ax, ["n20degC", "10degC", "25degC"]):
        u, _ = keep[(f, sel)]
        sl = slice(3600, 4200)
        a.plot(u.time_s[sl], u.voltage_v[sl], "k", lw=1, label="measured")
        for name, c in (("2rc", "tab:orange"), (sel, "tab:blue")):
            rm = ecm[(ecm.folder == f) & (ecm.test == "UDDS") & (ecm.model == name)].v_rmse_mv.iloc[0]
            a.plot(u.time_s[sl], keep[(f, name)][1][sl], color=c, lw=1, label=f"{name} (full-cycle RMSE {rm:.0f} mV)")
        a.set(title=f"UDDS (test cycle) at {NICE[f]}: 10 min window after 1 h, open loop", ylabel="V")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(FIG / "m2_ecm_drive_cycle.png", dpi=110)


if __name__ == "__main__":
    main()
