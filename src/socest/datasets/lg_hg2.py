"""Loader for the LG 18650HG2 dataset (Kollmeyer, Naguib, Skells, Mendeley Data V2,
2020, doi:10.17632/b5mj79w5w9.2, CC BY 4.0).

Per the dataset readme, each test is one .mat file whose name starts with the test
date and time, followed by the test series number and a descriptor, for example
"10-28-18_17.44 551_UDDS_25degC_LGHG2.mat", and whose columns include Time,
TimeStamp, Voltage, Current, Ah, Wh, Power and Battery_Temp_degC.

This loader does not assume the variable layout inside the .mat file: it searches
for a struct or table holding the documented column names. It converts current to
the package convention (i > 0 discharge) using a sign detected from a test whose
direction is known from its name, and fails loudly if detection is ambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = {
    "Time": ("time_s", "s"),
    "Voltage": ("voltage_v", "V"),
    "Current": ("current_a", "A"),
    "Ah": ("ah", "Ah"),
    "Wh": ("wh", "Wh"),
    "Power": ("power_w", "W"),
    "Battery_Temp_degC": ("temp_degc", "degC"),
}
REQUIRED = ("Time", "Voltage", "Current", "Battery_Temp_degC")

FOLDER_TEMPS = {"n20degC": -20.0, "n10degC": -10.0, "0degC": 0.0, "10degC": 10.0, "25degC": 25.0, "40degC": 40.0}

DRIVE_CYCLES = ("UDDS", "HWFET", "LA92", "US06") + tuple(f"Mixed{k}" for k in range(1, 9))
# Tests whose name fixes the current direction (used for sign detection).
KNOWN_DISCHARGE = ("C20DisCh", "Dis_0p5C", "Dis_2C", "Cap_1C")

_NAME_RE = re.compile(
    r"^(?P<stamp>\d{2}-\d{2}-\d{2}_?\d{2}\.\d{2})\s+(?P<series>\d+)_(?P<test>[A-Za-z0-9]+(?:_[0-9p]+C)?)"
)


@dataclass(frozen=True)
class TestFile:
    path: Path
    folder: str
    ambient_degc: float
    series: str
    test: str
    stamp: str

    @property
    def kind(self) -> str:
        if self.test in DRIVE_CYCLES:
            return "drive_cycle"
        if self.test.startswith("Charge"):
            return "charge"
        return {"HPPC": "hppc", "C20DisCh": "ocv_c20", "PausCycl": "pause_cycle"}.get(self.test, "discharge")


def parse_name(path: Path) -> TestFile:
    m = _NAME_RE.match(path.stem)
    if not m:
        raise ValueError(f"unrecognised LG HG2 file name: {path.name}")
    folder = path.parent.name
    return TestFile(path, folder, FOLDER_TEMPS.get(folder, float("nan")), m["series"], m["test"], m["stamp"])


def _find_columns(obj, depth=0):
    """Depth-first search for a container exposing the required column names."""
    if depth > 6 or obj is None:
        return None
    names = None
    if isinstance(obj, dict):
        names = obj.keys()
    elif hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
        names = obj.dtype.names
    if names is not None and all(k in names for k in REQUIRED):
        return obj
    children = []
    if isinstance(obj, dict):
        children = [v for k, v in obj.items() if not str(k).startswith("__")]
    elif hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
        children = [obj[n] for n in obj.dtype.names]
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        children = list(obj.ravel())
    for c in children:
        hit = _find_columns(c, depth + 1)
        if hit is not None:
            return hit
    return None


def _col(container, key):
    v = container[key]
    while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
        v = v.item()
    return np.asarray(v).ravel()


def read_mat(path: Path) -> dict:
    """Return raw columns as 1-D arrays, keyed by the dataset's column names."""
    path = Path(path)
    try:
        from scipy.io import loadmat

        raw = loadmat(path, squeeze_me=True, struct_as_record=True)
        container = _find_columns(raw)
        if container is None:
            raise KeyError(f"no struct with columns {REQUIRED} in {path.name}; keys: {list(raw)}")
        return {k: _col(container, k).astype(float) for k in RAW_COLUMNS if _has(container, k)}
    except (NotImplementedError, ValueError):  # MATLAB v7.3 files are HDF5
        import h5py

        with h5py.File(path, "r") as f:
            group = _find_h5_group(f)
            if group is None:
                raise KeyError(f"no HDF5 group with columns {REQUIRED} in {path.name}")
            return {k: np.asarray(group[k]).ravel().astype(float) for k in RAW_COLUMNS if k in group}


def _has(container, key):
    names = container.keys() if isinstance(container, dict) else container.dtype.names
    return key in names


def _find_h5_group(group):
    import h5py

    if all(k in group for k in REQUIRED):
        return group
    for v in group.values():
        if isinstance(v, h5py.Group):
            hit = _find_h5_group(v)
            if hit is not None:
                return hit
    return None


def detect_discharge_sign(files) -> int:
    """Return +1 if the raw data reports discharge as positive current, -1 if negative.

    Uses tests named as discharges (C/20, 0.5C, 1C, 2C): the dominant sign of
    current while it flows must agree across every such file.
    """
    signs = []
    for tf in files:
        if tf.test not in KNOWN_DISCHARGE:
            continue
        cur = read_mat(tf.path)["Current"]
        flowing = cur[np.abs(cur) > 0.05 * np.max(np.abs(cur))]
        if flowing.size == 0:
            continue
        frac_pos = np.mean(flowing > 0)
        if 0.1 < frac_pos < 0.9:
            raise ValueError(f"{tf.path.name}: mixed current sign in a discharge test ({frac_pos:.2f} positive)")
        signs.append(1 if frac_pos >= 0.9 else -1)
    if not signs:
        raise ValueError("no discharge tests found to detect the current sign")
    if len(set(signs)) != 1:
        raise ValueError(f"inconsistent discharge sign across files: {signs}")
    return signs[0]


def to_frame(tf: TestFile, discharge_sign: int) -> pd.DataFrame:
    raw = read_mat(tf.path)
    df = pd.DataFrame({RAW_COLUMNS[k][0]: v for k, v in raw.items()})
    df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
    # Package convention: i > 0 is discharge.
    df["current_a"] = df["current_a"] * discharge_sign
    if "power_w" in df:
        df["power_w"] = df["power_w"] * discharge_sign
    for k in ("ah", "wh"):
        if k in df:
            df[k] = df[k] * discharge_sign
    df.attrs.update(
        source_file=tf.path.name, folder=tf.folder, ambient_degc=tf.ambient_degc,
        series=tf.series, test=tf.test, kind=tf.kind,
    )
    return df


def list_files(raw_dir) -> list[TestFile]:
    raw_dir = Path(raw_dir)
    out = []
    for folder in FOLDER_TEMPS:
        for p in sorted((raw_dir / folder).glob("*.mat")):
            out.append(parse_name(p))
    return out
