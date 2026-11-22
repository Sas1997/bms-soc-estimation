"""Reproduce every result in this repository with one command.

    python run_all.py                 # download, prepare, identify, train, tune, evaluate
    python run_all.py --skip-train    # reuse the committed network weights in results/models/
    python run_all.py --skip-download # raw data already in data/raw/LG_HG2/

Steps (each is a standalone script and can be run on its own):
  1. data/download.py                  raw LG HG2 .mat files from Mendeley Data
  2. data/prepare.py                   Parquet + data/DATA_CARD.md
  3. experiments/m2_characterize.py    OCV and ECM variants from C/20 and HPPC
  4. experiments/m2_calibrate_slow.py  slow branch on the training drive cycles
  5. experiments/m2_select_ecm.py      model selection on the validation cycles
  6. experiments/m2_validate.py        open-loop voltage check, coulomb counting baseline
  7. experiments/m3_train.py           networks (skipped with --skip-train)
  8. experiments/m3_tune_ekf.py        EKF noise settings on the validation cycles
  9. experiments/m3_select_nn.py       network checkpoints on the validation cycles
 10. experiments/m4_evaluate.py        test-set metrics and robustness cases
 11. experiments/m4_cost.py            time per step and memory on this machine
 12. experiments/m4_figures.py         headline figure
 13. experiments/m6_balancing.py       passive vs active cell balancing (pack simulation)
 14. experiments/m7_smbus.py           smart-battery (SMBus) input path on an emulated pack

Edge demo files (run before step 8 and step 11, which use them):
  experiments/m5_export_replay.py      demo replay excerpt and capacity table
  experiments/m5_export_onnx.py        ONNX models of the selected networks
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Network configurations. All use seed 42 and one PyTorch thread.
TRAIN_RUNS = [
    ["--model", "transformer", "--name", "transformer"],
    ["--model", "transformer", "--name", "transformer_w600", "--window", "600", "--patch", "10"],
    ["--model", "transformer", "--name", "transformer_w600_aug", "--window", "600", "--patch", "10",
     "--noise", "0.005", "0.05", "0.5", "--bias", "0.05"],
    ["--model", "transformer", "--name", "transformer_causal", "--causal"],
    ["--model", "lstm", "--name", "lstm"],
    ["--model", "lstm", "--name", "lstm_aug", "--noise", "0.005", "0.05", "0.5", "--bias", "0.05"],
]


def run(*args):
    print(f"\n=== {' '.join(args)} ===", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    print(f"--- done in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--cost-label", default="laptop", help="label for the compute-cost table (for example raspberry-pi)")
    args = ap.parse_args()

    if not args.skip_download:
        run("data/download.py")
    run("data/prepare.py")
    for s in ("m2_characterize", "m2_calibrate_slow", "m2_select_ecm", "m2_validate", "m5_export_replay"):
        run(f"experiments/{s}.py")
    if not args.skip_train:
        for cfg in TRAIN_RUNS:
            run("experiments/m3_train.py", *cfg, "--threads", "1", "--epochs", "40")
    for s in ("m3_tune_ekf", "m3_select_nn", "m5_export_onnx", "m4_evaluate"):
        run(f"experiments/{s}.py")
    run("experiments/m4_cost.py", "--label", args.cost_label)
    run("experiments/m4_figures.py")
    run("experiments/m6_balancing.py")
    run("experiments/m7_smbus.py")
    print("\nAll results are in results/. Summary: results/tables/m4_summary.md, figure: results/figures/headline.png")


if __name__ == "__main__":
    main()
