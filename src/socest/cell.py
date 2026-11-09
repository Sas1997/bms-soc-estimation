"""Load the identified LG HG2 cell model (outputs of experiments/m2_characterize.py).

The model never uses 10 degC data: OCV, parameters and capacity are interpolated
across the other five temperatures, so the 10 degC test set is unseen for the
physics-based estimator, just as for the ML model.
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


def load_ocv(exclude=(HELD_OUT_FOLDER,)) -> OCVCurve:
    d = json.loads((CHAR / "ocv.json").read_text())
    keep = [k for k, f in enumerate(d["folders"]) if f not in exclude]
    return OCVCurve(d["soc"], np.array(d["ocv"])[keep], temps=np.array(d["temps"])[keep])


def capacity_fn(exclude=(HELD_OUT_FOLDER,)):
    """C/20 discharge capacity (Ah) as a function of temperature (linear interpolation, held at the ends)."""
    cap = pd.read_csv(PROC / "capacity.csv", index_col=0)
    cap = cap.drop(index=[f for f in exclude if f in cap.index])
    temps = np.array([FOLDER_TEMPS[f] for f in cap.index])
    o = np.argsort(temps)
    vals = cap["c20_discharge_ah"].to_numpy()[o]
    return lambda T: float(np.interp(T, temps[o], vals))


def load_model(n_rc: int = 2, capacity_ah: float | None = None, temp_for_capacity: float = 25.0) -> TheveninModel:
    params = ParamTable.from_json(CHAR / f"params_{n_rc}rc.json")
    cap = capacity_ah if capacity_ah is not None else capacity_fn()(temp_for_capacity)
    return TheveninModel(load_ocv(), params, cap)
