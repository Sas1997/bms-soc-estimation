"""Load an identified LG HG2 cell model (outputs of experiments/m2_characterize.py).

No model uses 10 degC data: OCV, parameters and capacity are interpolated across
the other five temperatures, so the 10 degC test set is unseen for the
physics-based estimators, just as for the ML models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets.lg_hg2 import FOLDER_TEMPS
from .ecm import TheveninModel
from .ocv import OCVCurve
from .params import ParamTable
from .split import HELD_OUT_FOLDER

ROOT = Path(__file__).resolve().parents[2]
CHAR = ROOT / "results" / "characterization"
PROC = ROOT / "data" / "processed" / "LG_HG2"


def variants() -> dict:
    return json.loads((CHAR / "variants.json").read_text())


def selected_variant() -> str:
    """Variant chosen on training and validation drive cycles (experiments/m2_select_ecm.py)."""
    return json.loads((CHAR / "selected.json").read_text())["variant"]


def load_ocv(file: str, exclude=(HELD_OUT_FOLDER,)) -> OCVCurve:
    d = json.loads((CHAR / file).read_text())
    keep = [k for k, f in enumerate(d["folders"]) if f not in exclude]
    return OCVCurve(d["soc"], np.array(d["ocv"])[keep], temps=np.array(d["temps"])[keep])


def capacity_fn(exclude=(HELD_OUT_FOLDER,)):
    """C/20 discharge capacity (Ah) as a function of temperature (linear interpolation, held at the ends)."""
    # The copy in results/ is committed, so the model loads without the processed dataset
    # (for example on a Raspberry Pi). It is written by experiments/m5_export_replay.py.
    src = CHAR / "capacity.csv" if (CHAR / "capacity.csv").exists() else PROC / "capacity.csv"
    cap = pd.read_csv(src, index_col=0)
    cap = cap.drop(index=[f for f in exclude if f in cap.index])
    temps = np.array([FOLDER_TEMPS[f] for f in cap.index])
    o = np.argsort(temps)
    vals = cap["c20_discharge_ah"].to_numpy()[o]
    return lambda T: float(np.interp(T, temps[o], vals))


def load_model(variant: str | None = None, capacity_ah: float | None = None, temp_for_capacity: float = 25.0) -> TheveninModel:
    """Build the model for a variant name from variants.json (default: the selected one).

    Callers pass the model the effective temperature:
        T_eff = effective_temperature(T_measured, model.thermal_gain)
    """
    v = variants()[variant or selected_variant()]
    cap = capacity_ah if capacity_ah is not None else capacity_fn()(temp_for_capacity)
    model = TheveninModel(load_ocv(v["ocv"]), ParamTable.from_json(CHAR / v["params"]), cap)
    # Parameters are looked up at socest.thermal.effective_temperature(T, model.thermal_gain).
    model.thermal_gain = float(v.get("thermal_gain", 0.0))
    return model
