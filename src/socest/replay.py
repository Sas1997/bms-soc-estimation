"""Small excerpt of the dataset used by the edge demo and the compute-cost benchmark.

demo/replay/ holds a few test cycles at 1 Hz (CSV), so a Raspberry Pi can run
the estimators without the raw dataset or the processing pipeline. The files
are written by experiments/m5_export_replay.py from data/processed.

Columns: time_s, voltage_v, current_a (positive = discharge), temp_degc, soc_ref.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "demo" / "replay"


def replay_names():
    return sorted(p.stem for p in REPLAY.glob("*.csv"))


def load_replay(name: str) -> pd.DataFrame:
    path = REPLAY / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; available: {', '.join(replay_names())}")
    return pd.read_csv(path)
