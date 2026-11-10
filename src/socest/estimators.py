"""One interface for every SoC estimator, so all of them see the same inputs.

Each estimator is a callable
    estimate(voltage, current, temp, soc0) -> soc array (N,)
on 1 Hz signals (V, A with positive = discharge, degC). soc0 is the initial SoC
handed to the estimator, which may be wrong on purpose; estimators without an
initial state (the neural networks) ignore it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cell import CHAR, capacity_fn, load_model, selected_variant
from .coulomb import coulomb_count
from .datasets.lg_hg2 import FOLDER_TEMPS
from .kalman import EKF
from .split import HELD_OUT_FOLDER
from .thermal import effective_temperature

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "results" / "models"


class CoulombCounter:
    name = "Coulomb counting"

    def __init__(self):
        self.cap_of = capacity_fn()

    def __call__(self, voltage, current, temp, soc0):
        return coulomb_count(current, 1.0, soc0, self.cap_of(temp[0]))


def voltage_sigma_fn(variant: str):
    """Model voltage error (V, 1 sigma) as a function of temperature: the median
    open-loop voltage RMSE of this variant on the TRAINING drive cycles at each
    model temperature (results/m2_ecm_selection.csv), linearly interpolated."""
    res = pd.read_csv(ROOT / "results" / "m2_ecm_selection.csv")
    sub = res[(res.variant == variant) & (res.split == "train") & (res.folder != HELD_OUT_FOLDER)]
    med = sub.groupby("folder").v_rmse_mv.median() / 1000.0
    temps = np.array([FOLDER_TEMPS[f] for f in med.index])
    o = np.argsort(temps)
    return lambda T: np.interp(T, temps[o], med.to_numpy()[o])


class EKFEstimator:
    """EKF on an identified cell model.

    Noise settings (see experiments/m3_tune_ekf.py for how they were chosen):
      R(T)   = (k_r * sigma_v(T))^2, sigma_v from voltage_sigma_fn: the measurement
               noise is dominated by model error, which grows in the cold
      Q      = diag(q_soc^2, q_v^2, ..., q_v^2) per 1 s step
      P0     = diag(p0_soc^2, p0_v^2, ...)
    """

    def __init__(self, variant: str | None = None, k_r: float = 1.0, q_soc: float = 1e-5, q_v: float = 1e-3,
                 p0_soc: float = 0.2, p0_v: float = 0.01, label: str | None = None):
        self.variant = variant or selected_variant()
        self.k_r, self.q_soc, self.q_v, self.p0_soc, self.p0_v = k_r, q_soc, q_v, p0_soc, p0_v
        self.cap_of = capacity_fn()
        self.sigma_v = voltage_sigma_fn(self.variant)
        self.name = label or f"EKF ({self.variant})"

    @classmethod
    def tuned(cls, variant: str | None = None, label: str | None = None):
        variant = variant or selected_variant()
        t = json.loads((CHAR / "ekf_tuning.json").read_text())[variant]
        return cls(variant, k_r=t["k_r"], q_soc=t["q_soc"], q_v=t["q_v"], label=label)

    def __call__(self, voltage, current, temp, soc0, return_all=False):
        model = load_model(self.variant, capacity_ah=self.cap_of(temp[0]))
        n = model.n_states
        x0 = np.zeros(n)
        x0[0] = soc0
        P0 = np.diag([self.p0_soc**2] + [self.p0_v**2] * (n - 1))
        Q = np.diag([self.q_soc**2] + [self.q_v**2] * (n - 1))
        R = (self.k_r * self.sigma_v(temp)) ** 2
        ekf = EKF(model, x0, P0, Q, float(R[0]))
        out = ekf.run(voltage, current, 1.0, effective_temperature(temp, model.thermal_gain), R=R)
        return out if return_all else out["soc"]


class NeuralEstimator:
    """Sequence-to-one network on the last `window` seconds of [V, I, T]."""

    def __init__(self, model_dir: str, label: str | None = None):
        from .ml.train import load

        self.model, self.norm, self.meta = load(MODELS / model_dir)
        self.name = label or model_dir

    def __call__(self, voltage, current, temp, soc0=None):
        from .ml.data import make_cycle
        from .ml.train import predict

        cyc = make_cycle(voltage, current, temp, np.zeros(len(voltage)))
        return predict(self.model, cyc, self.norm, self.meta["window"]).astype(float)


def selected_networks() -> dict:
    """Network checkpoints chosen on the validation cycles (experiments/m3_select_nn.py)."""
    return json.loads((MODELS / "selected.json").read_text())


def build_estimators():
    """The five estimators compared in the evaluation, in display order."""
    nn = selected_networks()
    return [
        CoulombCounter(),
        EKFEstimator.tuned("2rc", label="EKF, 2RC (HPPC only)"),
        EKFEstimator.tuned(None, label="EKF, 3RC (calibrated)"),
        NeuralEstimator(nn["lstm"], label="LSTM"),
        NeuralEstimator(nn["transformer"], label="Transformer"),
    ]
