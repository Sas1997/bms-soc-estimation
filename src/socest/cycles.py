"""Drive cycles at 1 Hz, cached on disk, for every estimator and experiment.

All estimators (coulomb counting, EKF, neural networks) see the same 1 Hz
signals produced by socest.resample.to_uniform, so comparisons are like for like.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .resample import to_uniform
from .split import load_index

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed" / "LG_HG2"
CACHE = ROOT / "data" / "processed" / "LG_HG2_1hz"


def load_cycle(row) -> pd.DataFrame:
    """1 Hz frame for one row of the split index (time_s, current_a, voltage_v, temp_degc, soc_ref)."""
    dest = CACHE / row["parquet"]
    if dest.exists():
        return pd.read_parquet(dest)
    u = to_uniform(pd.read_parquet(PROC / row["parquet"]), 1.0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    u.to_parquet(dest, index=False)
    return u


def load_split(*splits):
    """List of (row, frame) for the given split names (see socest.split.SPLITS)."""
    ix = load_index(PROC)
    return [(r, load_cycle(r)) for _, r in ix[ix["split"].isin(splits)].iterrows()]
