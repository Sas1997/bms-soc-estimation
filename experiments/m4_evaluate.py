"""M4: evaluate every estimator on the two test sets, with robustness cases.

Test sets (never used for fitting, calibration, tuning or selection):
  unseen cycle        UDDS, HWFET, LA92, US06 at -20, -10, 0, 25, 40 degC
  unseen temperature  every usable drive cycle at 10 degC

Cases:
  nominal             correct initial SoC, signals as logged
  initial SoC -20     estimator starts at 0.80 (true 1.00) with the cell at rest;
                      networks have no initial state
  late start -20      estimation starts 20 min into the cycle, under load, with the
                      estimator 20 points below the true SoC; networks start with an
                      empty input window
  sensor noise        white noise: 10 mV, 100 mA, 1 degC (1 sigma)
  current bias        +100 mA on the measured current

Metrics per cycle: RMSE, MAE and maximum absolute SoC error (percentage points);
for the wrong-initial-SoC case also the time after which the error stays within 5 points.

Writes results/m4_metrics.csv, results/m4_traces.npz, results/tables/m4_*.md.
Run: python experiments/m4_evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socest.cycles import load_split  # noqa: E402
from socest.estimators import build_estimators  # noqa: E402
from socest.metrics import Perturbation, convergence_time, soc_errors  # noqa: E402

CASES = [
    Perturbation("nominal"),
    Perturbation("initial SoC -20 pts", soc0_error=-0.20),
    Perturbation("late start, SoC -20 pts", soc0_error=-0.20, start_s=1200),
    Perturbation("sensor noise", v_noise_std=0.010, i_noise_std=0.100, t_noise_std=1.0),
    Perturbation("current bias +100 mA", i_bias=0.100),
]
TRACES = {("n20degC", "UDDS"), ("25degC", "UDDS"), ("10degC", "US06")}
TRACE_CASES = ("nominal", "late start, SoC -20 pts")
ORDER = ["n20degC", "n10degC", "0degC", "10degC", "25degC", "40degC"]
NICE = {"n20degC": "-20 °C", "n10degC": "-10 °C", "0degC": "0 °C", "10degC": "10 °C", "25degC": "25 °C", "40degC": "40 °C"}
SPLIT_NICE = {"test_unseen_cycle": "unseen cycle", "test_unseen_temp": "unseen temperature"}


def main():
    ests = build_estimators()
    rows, traces = [], {}
    for r, u in load_split("test_unseen_cycle", "test_unseen_temp"):
        for c_id, case in enumerate(CASES):
            k0 = case.start_s
            ref = u.soc_ref.to_numpy()[k0:]
            t = u.time_s.to_numpy()[k0:]
            v, i, T = case.apply(u.voltage_v.to_numpy()[k0:], u.current_a.to_numpy()[k0:], u.temp_degc.to_numpy()[k0:],
                                 seed=1000 + c_id)
            for est in ests:
                soc = est(v, i, T, ref[0] + case.soc0_error)
                m = {k: 100 * x for k, x in soc_errors(soc, ref).items()}
                conv = convergence_time(soc, ref, t, band=0.05) if case.soc0_error else np.nan
                rows.append({"split": r.split, "folder": r.folder, "test": r.test, "case": case.name,
                             "estimator": est.name, **m, "converge_s": conv})
                if (r.folder, r.test) in TRACES and case.name in TRACE_CASES:
                    traces[f"{r.folder}|{r.test}|{case.name}|{est.name}"] = soc.astype(np.float32)
            if (r.folder, r.test) in TRACES and case.name in TRACE_CASES:
                traces[f"{r.folder}|{r.test}|{case.name}|ref"] = ref.astype(np.float32)
        print(f"{r.folder}/{r.test} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "m4_metrics.csv", index=False)
    np.savez_compressed(ROOT / "results" / "m4_traces.npz", **traces)
    write_tables(res, [e.name for e in ests])


def write_tables(res, names):
    tab = ROOT / "results" / "tables"
    res = res.copy()
    res["set"] = res.split.map(SPLIT_NICE)
    nom = res[res.case == "nominal"]

    def per(sub, col):
        t = sub.pivot_table(index="estimator", columns=col, values="rmse", aggfunc="mean")
        return t.reindex(names)

    # headline: mean over cycles of each metric, nominal, both test sets
    head = nom.groupby(["estimator", "set"])[["rmse", "mae", "max_abs"]].mean().unstack("set").reindex(names)
    head.columns = [f"{m.upper().replace('MAX_ABS', 'MAX')} {s}" for m, s in head.columns]
    rob = res.pivot_table(index="estimator", columns=["set", "case"], values="rmse", aggfunc="mean").reindex(names)
    case_order = [c.name for c in CASES]
    by_temp = per(nom, "folder")
    by_temp = by_temp[[f for f in ORDER if f in by_temp.columns]].rename(columns=NICE)
    by_cycle = per(nom[nom.split == "test_unseen_cycle"], "test")
    worst = nom.groupby("estimator").max_abs.max().reindex(names)
    stateful = [n for n in names if n.startswith(("Coulomb", "EKF"))]
    late = res[res.case == "late start, SoC -20 pts"]
    conv = late.groupby("estimator").agg(median_s=("converge_s", "median"), p90_s=("converge_s", lambda x: x.quantile(0.9)),
                                         not_settled=("converge_s", lambda x: int(x.isna().sum()))).reindex(stateful)

    md = ["# M4 results: SoC error on the test sets (percentage points)", "",
          "Mean over test cycles. Unseen cycle: UDDS, HWFET, LA92, US06 at five temperatures (19 cycles). "
          "Unseen temperature: all usable cycles at 10 °C (10 cycles).", "",
          "## Nominal conditions", "", head.round(2).to_markdown(), "",
          "Coulomb counting is near-exact here by construction: the reference SoC is coulomb counting with a "
          "known full-charge start and the laboratory current sensor.", "",
          "## Robustness: RMSE by case", ""]
    for s in SPLIT_NICE.values():
        md += [f"### {s.capitalize()}", "", rob[s][case_order].round(2).to_markdown(), ""]
    md += ["Networks have no initial state, so their wrong-initial-SoC result at rest equals their nominal result. "
           "In the late-start case they begin with an empty input window.", "",
           "Late start: time after which the error stays within 5 points (seconds from the start of estimation; "
           "`not_settled` counts cycles where the error is still outside 5 points at the end):", "",
           conv.round(0).to_markdown(), "",
           "## Nominal RMSE by temperature (both test sets)", "", by_temp.round(2).to_markdown(), "",
           "## Nominal RMSE by drive cycle (unseen-cycle set, mean over temperatures)", "", by_cycle.round(2).to_markdown(), "",
           "## Largest single-cycle maximum error, nominal", "", worst.round(2).to_frame("max abs error").to_markdown(), ""]
    (tab / "m4_summary.md").write_text("\n".join(md))
    print(head.round(2).to_string())
    for s in SPLIT_NICE.values():
        print(s, "\n", rob[s][case_order].round(2).to_string())
    print(by_temp.round(2).to_string())
    print(conv.round(0).to_string())


if __name__ == "__main__":
    if "--tables-only" in sys.argv:      # rebuild the tables from results/m4_metrics.csv
        _res = pd.read_csv(ROOT / "results" / "m4_metrics.csv")
        write_tables(_res, list(dict.fromkeys(_res.estimator)))
    else:
        main()
