"""Compute cost of each estimator, measured sample by sample as on an embedded target.

Each estimator runs in its own Python process, so the peak memory figure is
that process alone (interpreter + libraries + model). PyTorch uses one thread.

    python experiments/m4_cost.py --label laptop        # all estimators
    python experiments/m4_cost.py --label raspberry-pi  # same command on the Pi

The networks are measured twice where PyTorch is installed: in PyTorch, and as
the exported ONNX models in ONNX Runtime (experiments/m5_export_onnx.py), which
is how they run on the edge device. Without PyTorch only the ONNX rows appear.
Without the processed dataset (for example on a Raspberry Pi) the input is the
25 degC UDDS replay file in demo/replay/, the same cycle.

Writes results/m4_cost_<label>.csv and rebuilds results/tables/m4_cost.md from
every results/m4_cost_*.csv present.

Reported per estimator:
  time per step   median wall time of one 1 Hz update (ms)
  peak memory     peak resident set size of the process (MB)
  model size      parameters (count) and storage on disk
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KEYS = ["coulomb", "ekf_2rc", "ekf_3rc", "lstm", "transformer", "lstm_onnx", "transformer_onnx"]


def cpu_name():
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/cpuinfo").read_text().splitlines() if ":" in line)
        info = {k.strip().lower(): v.strip() for k, v in info.items()}
        for key in ("model name", "model", "hardware"):     # x86: "model name"; Raspberry Pi: "Model"
            if info.get(key) and not info[key].isdigit():
                return info[key]
    except OSError:
        pass
    return platform.processor() or platform.machine()


def measure(key: str, n_steps: int):
    import numpy as np

    from socest.cell import CHAR, selected_variant, variants
    from socest.estimators import MODELS, EKFEstimator, NeuralEstimator, selected_networks
    from socest.streaming import StreamingCoulomb, StreamingEKF, StreamingNN

    try:
        from socest.cycles import load_split

        r, u = next((r, u) for r, u in load_split("test_unseen_cycle") if r.folder == "25degC" and r.test == "UDDS")
    except (FileNotFoundError, OSError):
        from socest.replay import load_replay

        u = load_replay("25degC_UDDS")
    v, i, T = (u[c].to_numpy()[:n_steps] for c in ("voltage_v", "current_a", "temp_degc"))
    params, size, name = 0, 0, key
    if key == "coulomb":
        s, name = StreamingCoulomb(2.78), "Coulomb counting"
    elif key.startswith("ekf"):
        variant = "2rc" if key == "ekf_2rc" else selected_variant()
        est = EKFEstimator.tuned(variant)
        s = StreamingEKF(est, temp0=float(T[0]))
        name = "EKF, 2RC (HPPC only)" if key == "ekf_2rc" else "EKF, 3RC (calibrated)"
        info = variants()[variant]
        files = [CHAR / info["params"], CHAR / info["ocv"]]
        size = sum(f.stat().st_size for f in files)
        tab = json.loads(files[0].read_text())
        params = sum(np.size(tab[k]) for k in ("r0", "r", "tau")) + np.size(json.loads(files[1].read_text())["ocv"])
    elif key.endswith("_onnx"):
        from socest.onnx_estimator import OnnxNetwork

        folder = selected_networks()[key.removesuffix("_onnx")]
        s = OnnxNetwork(folder, threads=1)
        name = ("LSTM" if key.startswith("lstm") else "Transformer") + " (ONNX Runtime)"
        params, size = s.meta["n_params"], (MODELS / folder / "model.onnx").stat().st_size
    else:
        import torch

        torch.set_num_threads(1)
        folder = selected_networks()[key]
        est = NeuralEstimator(folder)
        s = StreamingNN(est)
        name = ("LSTM" if key == "lstm" else "Transformer") + " (PyTorch)"
        params, size = est.meta["n_params"], (MODELS / folder / "best.pt").stat().st_size
    for k in range(20):                      # warm-up
        s.step(v[k], i[k], T[k])
    times = np.empty(n_steps - 20)
    for k in range(20, n_steps):
        t0 = time.perf_counter()
        s.step(v[k], i[k], T[k])
        times[k - 20] = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1024 if sys.platform != "darwin" else rss / 1024**2
    return {"estimator": name, "ms_per_step_median": 1000 * float(np.median(times)),
            "ms_per_step_p99": 1000 * float(np.percentile(times, 99)), "peak_rss_mb": rss_mb,
            "table_or_weight_values": int(params), "storage_kb": size / 1024}


def rebuild_table():
    import pandas as pd

    frames = [pd.read_csv(f) for f in sorted((ROOT / "results").glob("m4_cost_*.csv"))]
    if not frames:
        return
    md = ["# Compute cost per estimator", "",
          "One 1 Hz update, measured sample by sample; each estimator in its own Python process, PyTorch and "
          "ONNX Runtime on one thread. Peak memory is the resident set size of that process, so it includes the "
          "interpreter and libraries (PyTorch or ONNX Runtime for the networks, NumPy and pandas for all).", ""]
    for df in frames:
        d = df.iloc[0]
        md += [f"## {d.label}", "", f"{d.cpu}, {d.machine}, Python {d.python}, {int(d.steps)} steps.", "",
               df[["estimator", "ms_per_step_median", "ms_per_step_p99", "peak_rss_mb", "table_or_weight_values", "storage_kb"]]
               .rename(columns={"ms_per_step_median": "ms per step (median)", "ms_per_step_p99": "ms per step (99th pct)",
                                "peak_rss_mb": "peak memory (MB)", "table_or_weight_values": "stored values",
                                "storage_kb": "storage (kB)"}).round(3).to_markdown(index=False), ""]
    (ROOT / "results" / "tables" / "m4_cost.md").write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="laptop")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--one", choices=KEYS, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        print(json.dumps(measure(args.one, args.steps)))
        return
    import pandas as pd

    import importlib.util

    has_torch = importlib.util.find_spec("torch") is not None
    rows = []
    for key in KEYS:
        if key in ("lstm", "transformer") and not has_torch:
            print(f"{key}: PyTorch not installed, skipped (the ONNX rows cover the networks)", flush=True)
            continue
        out = subprocess.run([sys.executable, __file__, "--one", key, "--steps", str(args.steps)],
                             capture_output=True, text=True, check=True)
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df["label"], df["cpu"], df["machine"] = args.label, cpu_name(), platform.machine()
    df["python"], df["steps"] = platform.python_version(), args.steps
    df.to_csv(ROOT / "results" / f"m4_cost_{args.label}.csv", index=False)
    rebuild_table()


if __name__ == "__main__":
    main()
