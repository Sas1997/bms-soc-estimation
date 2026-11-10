"""Choose the EKF noise settings on the VALIDATION cycles.

The reference SoC is coulomb counting with a known start and the laboratory
current sensor, so with a correct initial SoC and a perfect sensor the best
"filter" is one that ignores voltage altogether. Tuning on that case alone would
switch the voltage feedback off. The objective therefore averages three cases
on each validation cycle:
    nominal            correct initial SoC, clean signals
    initial SoC -20    estimator starts at 0.80 instead of 1.00
    current bias       +50 mA added to the measured current
Score = mean over cases of the mean SoC RMSE over the validation cycles.

Grid (per 1 s step):
    k_r    scale on the temperature-dependent model voltage error sigma_v(T)
    q_soc  process noise on SoC (fraction)
    q_v    process noise on each RC voltage (V)
Fixed: P0 = diag(0.2^2, 0.01^2, ...).

Writes results/characterization/ekf_tuning.json and results/tables/m3_ekf_tuning.md.
Run: python experiments/m3_tune_ekf.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socest.cell import CHAR, selected_variant  # noqa: E402
from socest.cycles import load_split  # noqa: E402
from socest.estimators import EKFEstimator  # noqa: E402
from socest.metrics import Perturbation, soc_errors  # noqa: E402

CASES = [Perturbation("nominal"), Perturbation("initial SoC -20 pts", soc0_error=-0.20),
         Perturbation("current bias +50 mA", i_bias=0.05)]
GRID = {"k_r": [0.5, 1.0, 2.0, 4.0, 8.0], "q_soc": [1e-6, 1e-5, 1e-4, 1e-3], "q_v": [1e-4, 1e-3, 1e-2]}


def score(est, data):
    out = {}
    for case in CASES:
        errs = []
        for r, u in data:
            v, i, T = case.apply(u.voltage_v.to_numpy(), u.current_a.to_numpy(), u.temp_degc.to_numpy(), seed=0)
            soc = est(v, i, T, 1.0 + case.soc0_error)
            errs.append(100 * soc_errors(soc, u.soc_ref)["rmse"])
        out[case.name] = float(np.mean(errs))
    out["score"] = float(np.mean(list(out.values())))
    return out


def main():
    data = load_split("val")
    variants = ["2rc", selected_variant()]
    tuning_edge = {}
    tuning, md = {}, ["# EKF noise tuning on the validation cycles", "", __doc__.split("Writes")[0].split("\n\n", 1)[1].strip(), ""]
    for variant in variants:
        rows = []
        for k_r, q_soc, q_v in itertools.product(*GRID.values()):
            s = score(EKFEstimator(variant, k_r=k_r, q_soc=q_soc, q_v=q_v), data)
            rows.append({"k_r": k_r, "q_soc": q_soc, "q_v": q_v, **s})
            print(variant, rows[-1], flush=True)
        tab = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
        best = tab.iloc[0]
        tuning[variant] = {"k_r": float(best.k_r), "q_soc": float(best.q_soc), "q_v": float(best.q_v),
                           "val_score_pts": round(float(best.score), 3)}
        show = tab.head(8).round(3).astype({"q_soc": str, "q_v": str})
        show["q_soc"], show["q_v"] = tab.head(8).q_soc.map("{:g}".format), tab.head(8).q_v.map("{:g}".format)
        edge = [k for k, vals in GRID.items() if best[k] in (min(vals), max(vals))]
        md += [f"## Model `{variant}`", "", "SoC RMSE in percentage points, best eight settings:", "",
               show.to_markdown(index=False), "",
               f"Worst setting on the grid scores {tab.score.max():.2f}. "
               + (f"The best setting is at the edge of the grid for: {', '.join(edge)}." if edge
                  else "The best setting is inside the grid for every parameter."), ""]
        tuning_edge[variant] = edge
    (CHAR / "ekf_tuning.json").write_text(json.dumps(tuning, indent=1))
    (ROOT / "results" / "tables" / "m3_ekf_tuning.md").write_text("\n".join(md))
    print(json.dumps(tuning, indent=1))


if __name__ == "__main__":
    main()
