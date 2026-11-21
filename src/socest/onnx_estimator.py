"""Neural SoC estimators run with ONNX Runtime instead of PyTorch.

For edge devices: ONNX Runtime is a small CPU-only library (about 23 MB on
Linux ARM64), while the PyPI build of PyTorch for Linux ARM64 is several
hundred MB and pulls CUDA libraries a Raspberry Pi cannot use. The ONNX files
are exported from the trained PyTorch weights by experiments/m5_export_onnx.py,
which checks that both give the same SoC.

Each model folder in results/models/ holds model.onnx (input "x", shape
(batch, window, 3), normalised [V, I, T]; output "soc", shape (batch,)) and
meta.json (window length and normalisation statistics from the training cycles).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "results" / "models"


class OnnxNetwork:
    """One exported network: batch prediction over a cycle, or one step at a time."""

    def __init__(self, model_dir: str, label: str | None = None, threads: int = 1):
        import onnxruntime as ort

        d = MODELS / model_dir
        self.meta = json.loads((d / "meta.json").read_text())
        self.window = int(self.meta["window"])
        self.mean = np.asarray(self.meta["normalizer"]["mean"], dtype=np.float32)
        self.std = np.asarray(self.meta["normalizer"]["std"], dtype=np.float32)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(d / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self.name = label or model_dir
        self.buf = None

    def _norm(self, x):
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def __call__(self, voltage, current, temp, soc0=None, batch: int = 512):
        """SoC for every sample of a cycle (same windowing as socest.ml.data.WindowDataset)."""
        x = self._norm(np.stack([voltage, current, temp], axis=1))
        x = np.concatenate([np.repeat(x[:1], self.window - 1, axis=0), x], axis=0)
        n = len(voltage)
        win = np.lib.stride_tricks.sliding_window_view(x, (self.window, 3))[:, 0]   # (n, window, 3)
        out = np.empty(n, dtype=np.float32)
        for a in range(0, n, batch):
            out[a : a + batch] = self.session.run(None, {"x": np.ascontiguousarray(win[a : a + batch])})[0]
        return out.astype(float)

    def reset(self):
        self.buf = None

    def step(self, voltage, current, temp):
        """Feed one 1 Hz sample, return the SoC estimate for it."""
        x = self._norm([voltage, current, temp])
        if self.buf is None:
            self.buf = np.repeat(x[None], self.window, axis=0)   # start-up: pad with the first sample
        else:
            self.buf[:-1] = self.buf[1:]
            self.buf[-1] = x
        return float(self.session.run(None, {"x": self.buf[None]})[0][0])
