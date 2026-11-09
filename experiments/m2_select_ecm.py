"""Score the cell model variants on drive cycles and select one on the VALIDATION cycles.

Variants come from experiments/m2_characterize.py (C/20 and HPPC only) and
experiments/m2_calibrate_slow.py (slow branch calibrated on the training drive
cycles). Each one predicts terminal voltage open loop (from current and
temperature, starting at SoC = 1) at 1 Hz.

Selection uses the validation cycles only (Mixed7 and Mixed8), which no variant
was fitted on. Score: voltage RMSE per cycle, median per temperature, then the
mean over the four temperatures that have validation cycles (40 degC has none,
see data/DATA_CARD.md). Training-cycle results are reported for reference.
Test cycles and all 10 degC data are not touched.

Writes results/characterization/selected.json, results/tables/m2_ecm_selection.md
and results/m2_ecm_selection.csv.
Run: python experiments/m2_select_ecm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socest.cell import CHAR, capacity_fn, load_model, variants  # noqa: E402
from socest.cycles import load_split  # noqa: E402
from socest.thermal import effective_temperature  # noqa: E402

ORDER = ["n20degC", "n10degC", "0degC", "25degC", "40degC"]


def main():
    cap_of = capacity_fn()
    data = load_split("train", "val")
    rows = []
    for name in variants():
        for r, u in data:
            i, v, T = u.current_a.to_numpy(), u.voltage_v.to_numpy(), u.temp_degc.to_numpy()
            m = load_model(name, capacity_ah=cap_of(T[0]))
            vs, _ = m.simulate(i, 1.0, effective_temperature(T, m.thermal_gain), soc0=1.0)
            e = 1000 * (vs - v)
            rows.append({"variant": name, "split": r.split, "folder": r.folder, "test": r.test,
                         "v_rmse_mv": np.sqrt(np.mean(e**2)), "bias_mv": e.mean()})
        print(f"{name} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "m2_ecm_selection.csv", index=False)

    names = list(variants())

    def table(split, col="v_rmse_mv"):
        t = res[res.split == split].pivot_table(index="variant", columns="folder", values=col, aggfunc="median")
        return t.reindex(index=names, columns=[f for f in ORDER if f in t.columns])

    val, train, bias = table("val"), table("train"), table("val", "bias_mv")
    val["score"] = val.mean(axis=1)
    train["mean"] = train.mean(axis=1)
    best = val["score"].idxmin()
    (CHAR / "selected.json").write_text(json.dumps(
        {"variant": best, "criterion": "mean over temperatures of the median open-loop voltage RMSE on validation drive cycles",
         "score_mv": round(float(val.loc[best, "score"]), 2)}, indent=1))

    desc = pd.Series({n: v["description"] for n, v in variants().items()}, name="model")
    md = ["# Cell model selection", "",
          "Open-loop voltage prediction at 1 Hz, median voltage RMSE per temperature (mV). "
          "No test cycle and no 10 °C data are used.", "",
          "## Validation cycles (Mixed7, Mixed8): used for selection", "",
          "Score = mean of the temperature columns. 40 °C has no validation cycles.", "",
          val.round(1).join(desc).to_markdown(), "",
          "Median bias on the validation cycles, model minus measured (mV):", "", bias.round(1).to_markdown(), "",
          "## Training cycles (Mixed1 to Mixed6): reference only", "",
          "Variants ending in `_dc` had their slow branch calibrated on these cycles.", "",
          train.round(1).to_markdown(), "",
          f"Selected: `{best}` ({desc[best]}).", ""]
    (ROOT / "results" / "tables" / "m2_ecm_selection.md").write_text("\n".join(md))
    print("VAL\n" + val.round(1).to_string())
    print("VAL bias\n" + bias.round(1).to_string())
    print("TRAIN\n" + train.round(1).to_string())
    print("selected:", best)


if __name__ == "__main__":
    main()
