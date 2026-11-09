"""Train a neural SoC estimator on the training cycles; early stopping on the validation cycles.

Examples:
    python experiments/m3_train.py --model transformer --name transformer
    python experiments/m3_train.py --model lstm --name lstm
Outputs go to results/models/<name>/ (best.pt, meta.json).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from socest.cycles import load_split  # noqa: E402
from socest.ml.data import make_cycle  # noqa: E402
from socest.ml.train import train  # noqa: E402


def as_cycles(pairs):
    return [make_cycle(u.voltage_v.to_numpy(), u.current_a.to_numpy(), u.temp_degc.to_numpy(), u.soc_ref.to_numpy(),
                       name=f"{r.folder}/{r.test}") for r, u in pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="transformer", choices=["transformer", "lstm"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--patch", type=int, default=5)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--noise", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("V", "A", "degC"))
    ap.add_argument("--bias", type=float, default=0.0, help="std of per-window current offset, A")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    tr, va = as_cycles(load_split("train")), as_cycles(load_split("val"))
    print(f"train {len(tr)} cycles, {sum(len(c['y']) for c in tr)} samples; val {len(va)} cycles", flush=True)
    kw = {"patch": args.patch} if args.model == "transformer" else {}
    meta = train(tr, va, ROOT / "results" / "models" / args.name, model_name=args.model, model_kwargs=kw,
                 window=args.window, stride=args.stride, batch_size=args.batch_size, lr=args.lr,
                 max_epochs=args.epochs, patience=args.patience, seed=args.seed,
                 noise_std=tuple(args.noise), current_bias_std=args.bias)
    print(f"best val RMSE {100 * meta['best_val_rmse']:.2f} pts at epoch {meta['best_epoch']}, "
          f"{meta['n_params']} params, {meta['train_seconds']:.0f} s")


if __name__ == "__main__":
    main()
