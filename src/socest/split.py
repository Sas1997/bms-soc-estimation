"""The single train/validation/test split used by every experiment.

Split by drive cycle and by temperature, never by time step, so no test sample
shares a cycle (or, for the unseen-temperature set, a temperature) with training.

  train                 Mixed1-6   at -20, -10, 0, 25, 40 degC
  val                   Mixed7-8   at -20, -10, 0, 25, 40 degC
  test_unseen_cycle     UDDS, HWFET, LA92, US06 at -20, -10, 0, 25, 40 degC
  test_unseen_temp      every drive cycle at 10 degC (never seen in training)

Excluded tests (see data/DATA_CARD.md) are dropped before splitting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HELD_OUT_FOLDER = "10degC"
TRAIN_MIXED = {f"Mixed{k}" for k in range(1, 7)}
VAL_MIXED = {"Mixed7", "Mixed8"}
STANDARD = {"UDDS", "HWFET", "LA92", "US06"}
SPLITS = ("train", "val", "test_unseen_cycle", "test_unseen_temp")


def assign(folder: str, test: str) -> str:
    if folder == HELD_OUT_FOLDER:
        return "test_unseen_temp"
    if test in TRAIN_MIXED:
        return "train"
    if test in VAL_MIXED:
        return "val"
    if test in STANDARD:
        return "test_unseen_cycle"
    raise ValueError(f"no split rule for {folder}/{test}")


def load_index(processed_dir) -> pd.DataFrame:
    """Usable drive cycles with a 'split' column."""
    ix = pd.read_csv(Path(processed_dir) / "index.csv")
    dc = ix[(ix["kind"] == "drive_cycle") & ~ix["excluded"]].copy()
    dc["split"] = [assign(f, t) for f, t in zip(dc["folder"], dc["test"])]
    return dc.reset_index(drop=True)
