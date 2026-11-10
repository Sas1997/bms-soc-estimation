"""Pick one LSTM and one Transformer checkpoint on the VALIDATION cycles.

Uses the same three cases as the EKF tuning (experiments/m3_tune_ekf.py), so both
families are selected on the same objective. A network has no initial state, so
its "initial SoC -20" result equals its nominal result.

Candidates are every folder under results/models/ whose meta.json names that model type.
Writes results/models/selected.json and results/tables/m3_nn_selection.md.
Run: python experiments/m3_select_nn.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from m3_tune_ekf import score  # noqa: E402
from socest.cycles import load_split  # noqa: E402
from socest.estimators import MODELS, NeuralEstimator  # noqa: E402


def main():
    data = load_split("val")
    rows = []
    for d in sorted(p for p in MODELS.iterdir() if (p / "meta.json").exists()):
        est = NeuralEstimator(d.name)
        m = est.meta
        rows.append({"checkpoint": d.name, "type": m["model_name"], "window_s": m["window"], "parameters": m["n_params"],
                     "augmented": bool(any(m.get("noise_std", [0])) or m.get("current_bias_std", 0)),
                     "epochs_run": len(m["history"]), **score(est, data)})
        print(rows[-1], flush=True)
    tab = pd.DataFrame(rows).sort_values(["type", "score"]).reset_index(drop=True)
    sel = {t: g.iloc[0].checkpoint for t, g in tab.groupby("type")}
    (MODELS / "selected.json").write_text(json.dumps(sel, indent=1))
    md = ["# Neural network selection on the validation cycles", "",
          "SoC RMSE in percentage points, mean over the validation cycles. Score = mean of the three cases. "
          "Augmented checkpoints were trained with added sensor noise (5 mV, 50 mA, 0.5 °C) and a random "
          "current offset (50 mA standard deviation).", "",
          tab.round(3).to_markdown(index=False), "",
          "Selected: " + ", ".join(f"{t}: `{c}`" for t, c in sel.items()) + ".", ""]
    (ROOT / "results" / "tables" / "m3_nn_selection.md").write_text("\n".join(md))
    print(tab.round(3).to_string(index=False))
    print(sel)


if __name__ == "__main__":
    main()
