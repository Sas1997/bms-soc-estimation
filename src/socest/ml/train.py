"""Training loop with fixed seeds, early stopping on validation RMSE, and checkpointing."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Normalizer, WindowDataset
from .models import build_model, n_params


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def predict(model, cycle, normalizer, window, batch_size=4096, device="cpu"):
    """SoC estimate for every step of one cycle."""
    ds = WindowDataset([cycle], normalizer, window, stride=1)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    out = [model(x.to(device)).cpu() for x, _ in dl]
    return torch.cat(out).numpy()


def evaluate_rmse(model, cycles, normalizer, window, device):
    errs = []
    for c in cycles:
        pred = predict(model, c, normalizer, window, device=device)
        errs.append((pred - c["y"]) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(errs))))


def train(
    train_cycles,
    val_cycles,
    out_dir,
    model_name="transformer",
    model_kwargs=None,
    window=300,
    stride=5,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=60,
    patience=8,
    seed=42,
    device=None,
    noise_std=(0.0, 0.0, 0.0),
    current_bias_std=0.0,
):
    """Train one estimator. Returns a dict summary; writes best.pt and meta.json.

    Optional training-time augmentation, in physical units, applied per batch:
      noise_std        white noise std on (V, A, degC)
      current_bias_std std of a per-window constant current offset (A)
    Validation data are never augmented.
    """
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_kwargs = dict(model_kwargs or {})
    if model_name == "transformer":
        model_kwargs.setdefault("window", window)

    norm = Normalizer.fit(train_cycles)
    ds = WindowDataset(train_cycles, norm, window, stride=stride)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)

    # augmentation scales in normalised units
    aug_noise = torch.tensor(np.asarray(noise_std) / norm.std, dtype=torch.float32, device=device)
    aug_bias = float(current_bias_std / norm.std[1])

    model = build_model(model_name, **model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = torch.nn.MSELoss()

    best, best_epoch, history = float("inf"), -1, []
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        total = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            if any(noise_std) or current_bias_std:
                x = x + torch.randn_like(x) * aug_noise
                if current_bias_std:
                    x[:, :, 1] += torch.randn(x.shape[0], 1, device=device) * aug_bias
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        val = evaluate_rmse(model, val_cycles, norm, window, device)
        history.append({"epoch": epoch, "train_rmse": (total / len(ds)) ** 0.5, "val_rmse": val})
        print(f"epoch {epoch:3d}  train {history[-1]['train_rmse']:.4f}  val {val:.4f}", flush=True)
        if val < best:
            best, best_epoch = val, epoch
            torch.save(model.state_dict(), out_dir / "best.pt")
        elif epoch - best_epoch >= patience:
            break

    meta = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "window": window,
        "stride": stride,
        "seed": seed,
        "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
        "noise_std": list(noise_std), "current_bias_std": current_bias_std,
        "n_params": n_params(model),
        "best_val_rmse": best,
        "best_epoch": best_epoch,
        "train_seconds": time.time() - t0,
        "device": device,
        "normalizer": norm.to_dict(),
        "history": history,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load(out_dir, device="cpu"):
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    model = build_model(meta["model_name"], **meta["model_kwargs"])
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    model.eval()
    return model, Normalizer.from_dict(meta["normalizer"]), meta
