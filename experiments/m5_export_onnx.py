"""Export the selected networks to ONNX for edge deployment, and check them.

For each network in results/models/selected.json this writes
results/models/<name>/model.onnx and compares ONNX Runtime with PyTorch on
every sample of one test cycle (25 degC UDDS, or the demo replay file when the
processed data is absent). The export is refused if the largest difference
exceeds 1e-4 SoC (0.01 points).

Writes results/tables/m5_onnx_export.md.
Run: python experiments/m5_export_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from socest.estimators import MODELS, NeuralEstimator, selected_networks  # noqa: E402
from socest.onnx_estimator import OnnxNetwork  # noqa: E402
from socest.replay import load_replay  # noqa: E402

TOL = 1e-4
OPSET = 18


def export(folder: str):
    est = NeuralEstimator(folder)
    model = est.model.eval()
    x = torch.zeros(2, est.meta["window"], 3)        # batch 2: dynamo would specialise a size-1 dimension
    path = MODELS / folder / "model.onnx"
    torch.onnx.export(model, (x,), str(path), input_names=["x"], output_names=["soc"],
                      dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=4096)}},
                      opset_version=OPSET, dynamo=True, external_data=False)
    return est, path


def main():
    u = load_replay("25degC_UDDS")
    v, i, T = (u[c].to_numpy() for c in ("voltage_v", "current_a", "temp_degc"))
    rows = []
    for kind, folder in selected_networks().items():
        est, path = export(folder)
        torch.set_num_threads(1)
        ref = est(v, i, T)
        onx = OnnxNetwork(folder)
        got = onx(v, i, T)
        step = np.array([onx.step(a, b, c) for a, b, c in zip(v[:600], i[:600], T[:600])])
        d_batch = float(np.max(np.abs(got - ref)))
        d_step = float(np.max(np.abs(step - ref[:600])))
        rows.append((kind, folder, path.stat().st_size / 1024, d_batch, d_step))
        print(f"{kind}: {path.relative_to(ROOT)} {path.stat().st_size / 1024:.0f} kB, "
              f"max |ONNX - PyTorch| = {d_batch:.2e} (batch), {d_step:.2e} (step by step)", flush=True)
        if max(d_batch, d_step) > TOL:
            sys.exit(f"{folder}: ONNX output differs from PyTorch by more than {TOL}")
    md = ["# ONNX export of the selected networks", "",
          f"Exported with torch.onnx (opset {OPSET}) and checked with ONNX Runtime on every sample of the "
          f"25 °C UDDS test cycle ({len(v)} samples at 1 Hz). Step-by-step check: first 600 samples.", "",
          "| Network | Checkpoint | ONNX file (kB) | Max SoC difference, batch | Max SoC difference, step by step |",
          "|:--|:--|--:|--:|--:|"]
    md += [f"| {k} | `{f}` | {s:.0f} | {a:.1e} | {b:.1e} |" for k, f, s, a, b in rows]
    (ROOT / "results" / "tables" / "m5_onnx_export.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
