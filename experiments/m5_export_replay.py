"""Write the demo replay files (demo/replay/*.csv) from the processed 1 Hz test cycles.

The files are a small excerpt of the LG 18650HG2 dataset (Kollmeyer, Naguib,
Skells, Mendeley Data V2, 2020, doi:10.17632/b5mj79w5w9.2), redistributed under
its CC BY 4.0 licence. Changes from the original: resampled to 1 Hz, current
sign changed to positive = discharge, reference SoC added (see data/DATA_CARD.md).

Only TEST cycles are included, so the demo shows estimators on data they were
never fitted or tuned on. Also copies the capacity table used by the cell model.
Run: python experiments/m5_export_replay.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socest.cycles import PROC, load_split  # noqa: E402
from socest.replay import REPLAY  # noqa: E402

CYCLES = [("25degC", "UDDS"), ("n20degC", "UDDS"), ("10degC", "US06")]
COLS = ["time_s", "voltage_v", "current_a", "temp_degc", "soc_ref"]
DECIMALS = {"time_s": 0, "voltage_v": 5, "current_a": 5, "temp_degc": 3, "soc_ref": 6}

README = """# Replay data for the edge demo

A small excerpt of the LG 18650HG2 dataset: P. Kollmeyer, M. Naguib, M. Skells,
*LG 18650HG2 Li-ion Battery Data*, Mendeley Data, V2, 2020,
doi:10.17632/b5mj79w5w9.2, licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).

Changes from the original files: resampled to 1 Hz with charge preserved,
current sign changed to positive = discharge, and a reference SoC column added
(tester Ah counter normalised by the C/20 capacity at that temperature). See
data/DATA_CARD.md in this repository.

All three are test cycles: no estimator in this repository was fitted, tuned
or selected on them. 10 °C is the held-out temperature.

| File | Ambient | Cycle | Samples (1 Hz) |
|:--|:--|:--|--:|
"""


def main():
    REPLAY.mkdir(parents=True, exist_ok=True)
    pairs = {(r.folder, r.test): u for r, u in load_split("test_unseen_cycle", "test_unseen_temp")}
    lines = [README]
    for folder, test in CYCLES:
        u = pairs[(folder, test)][COLS].copy()
        for c, d in DECIMALS.items():
            u[c] = u[c].round(d)
        u["time_s"] = u["time_s"].astype(int)
        name = f"{folder}_{test}"
        u.to_csv(REPLAY / f"{name}.csv", index=False)
        temp = folder.replace("n", "-").replace("degC", " °C")
        lines.append(f"| `{name}.csv` | {temp} | {test} | {len(u)} |")
        print(f"{name}: {len(u)} samples, {(REPLAY / f'{name}.csv').stat().st_size / 1024:.0f} kB")
    (REPLAY / "README.md").write_text("\n".join(lines) + "\n")
    shutil.copy(PROC / "capacity.csv", ROOT / "results" / "characterization" / "capacity.csv")


if __name__ == "__main__":
    main()
