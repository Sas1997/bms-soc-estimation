"""Sliding-window datasets for sequence-to-one SoC estimation.

Each sample is a window of the last L measurements [V, I, T] ending at step k;
the target is the reference SoC at step k. Windows never cross cycle
boundaries, and normalisation statistics come from the training cycles only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

FEATURES = ("voltage", "current", "temp")


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, cycles):
        x = np.concatenate([c["x"] for c in cycles], axis=0)
        return cls(mean=x.mean(axis=0), std=x.std(axis=0) + 1e-8)

    def __call__(self, x):
        return (x - self.mean) / self.std

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "features": list(FEATURES)}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["mean"]), np.asarray(d["std"]))


def make_cycle(voltage, current, temp, soc, **meta):
    """Pack one cycle. Arrays must share a uniform time step."""
    x = np.stack([voltage, current, temp], axis=1).astype(np.float32)
    return {"x": x, "y": np.asarray(soc, dtype=np.float32), **meta}


class WindowDataset(Dataset):
    """All windows of length ``window`` with targets at the window end.

    stride > 1 subsamples window end points (training only). The first
    window-1 steps of each cycle are left-padded with the first sample so every
    step gets an estimate; this mirrors a device that has just started up.
    """

    def __init__(self, cycles, normalizer: Normalizer, window: int, stride: int = 1):
        self.window = window
        self.xs, self.ys, self.index = [], [], []
        for c_id, c in enumerate(cycles):
            x = normalizer(c["x"]).astype(np.float32)
            pad = np.repeat(x[:1], window - 1, axis=0)
            self.xs.append(np.concatenate([pad, x], axis=0))
            self.ys.append(c["y"])
            self.index.extend((c_id, k) for k in range(0, len(c["y"]), stride))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        c_id, k = self.index[idx]
        x = self.xs[c_id][k : k + self.window]
        return torch.from_numpy(x), torch.tensor(self.ys[c_id][k])
