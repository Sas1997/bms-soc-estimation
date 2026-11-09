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
    "Power": ("power_w", "W"),
    "Battery_Temp_degC": ("temp_degc", "degC"),
}
REQUIRED = ("Time", "Voltage", "Current", "Battery_Temp_degC")
# Not loaded: "Wh" is stored as uint8 with a single constant value (3) in the files
# inspected (for example 25degC/551_UDDS), so it carries no information.
# "TimeStamp" is a wall-clock string; "Time" is used instead.

# Tests excluded from all experiments, with the evidence found in the files.
# Keys are (folder, series, test). See data/DATA_CARD.md for the numbers.
EXCLUDE = {
    ("25degC", "551", "HWFET"): "no load: 10 min of rest only, current is zero throughout",
    ("40degC", "562", "Mixed4"): "not at 40 degC (cell 22.9 to 23.5 degC), truncated (0.32 h); preceding charge stopped at 3.39 V",
    ("40degC", "562", "Mixed5"): "not at 40 degC (cell 23.9 to 24.5 degC), truncated (0.36 h)",
    ("40degC", "562", "Mixed6"): "not at 40 degC (cell 23.2 to 25.2 degC), truncated (0.33 h)",
    ("40degC", "562", "Mixed7"): "not at 40 degC (cell 22.7 to 24.4 degC), truncated (0.22 h)",
    ("40degC", "562", "Mixed8"): "not at 40 degC (cell 22.8 to 24.0 degC), truncated (0.40 h)",
    ("n10degC", "602", "Mixed4"): "416 s logging gap under load at t = 387 s (0.29 Ah passed during the gap per the tester Ah counter)",
    ("10degC", "582", "LA92"): "full-charge start uncertain: re-run 5 days after its series with no charge file in series 582, "
                               "3 logging restarts, rested start 4.170 V vs 4.184 to 4.187 V for the other rested 10 degC starts",
}

# Drive cycles are assumed to start fully charged. Checked per cycle: the file just
# before it in the same folder must be a charge that reached the 4.2 V CV limit.
# If no charge file precedes it, the cycle must start at rest (|I| < 0.1 A) at or
# above FALLBACK_REST_V. A start-voltage threshold alone is NOT used: Mixed4 and
# Mixed8 begin under load at t = 0, so their first voltage is a loaded voltage.
CHARGE_CV_LIMIT_V = 4.195
FALLBACK_REST_V = 4.17

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
    def when(self):
        """Test start from the file name stamp (month-day-year as written)."""
        from datetime import datetime

        m = re.match(r"(\d{2})-(\d{2})-(\d{2})_?(\d{2})\.(\d{2})", self.stamp)
        mo, d, y, hh, mm = map(int, m.groups())
        return datetime(2000 + y, mo, d, hh, mm)

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


def first_sustained_segment(time, current, min_duration_s=60.0, min_current_a=0.02):
    """Return (start, end, sign) of the first run of same-sign current lasting at
    least ``min_duration_s``, or None."""
    s = np.sign(np.where(np.abs(current) > min_current_a, current, 0.0))
    edges = np.flatnonzero(np.diff(s)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges - 1, len(s) - 1]
    for a, b in zip(starts, ends):
        if s[a] != 0 and time[b] - time[a] >= min_duration_s:
            return int(a), int(b), int(s[a])
    return None


def detect_discharge_sign(files) -> int:
    """Return +1 if the raw data reports discharge as positive current, -1 if negative.

    Uses tests named as discharges (C/20, 0.5C, 1C, 2C). Some of these files also
    contain a recharge after the discharge, so the sign is taken from the FIRST
    sustained current segment, and that segment must lower the voltage by at
    least 0.3 V (a physical check that it really is the discharge). The sign must
    agree across every such file.
    """
    signs = []
    for tf in files:
        if tf.test not in KNOWN_DISCHARGE:
            continue
        c = read_mat(tf.path)
        seg = first_sustained_segment(c["Time"], c["Current"])
        if seg is None:
            continue
        i0, i1, sign = seg
        dv = c["Voltage"][i1] - c["Voltage"][i0]
        if dv > -0.3:
            raise ValueError(f"{tf.path.name}: first sustained segment does not discharge the cell (dV={dv:+.3f} V)")
        signs.append(sign)
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


def repair_time(df: pd.DataFrame):
    """Make time strictly usable: drop NaN rows, and if the time column jumps
    backwards (logging restarts), keep only the part after the last restart.

    Returns (df, note). The tester Ah counter keeps running through restarts, so
    the reference SoC remains continuous; it is computed from the Ah column.
    """
    notes = []
    n_nan = int(df[["voltage_v", "current_a", "temp_degc"]].isna().any(axis=1).sum())
    if n_nan:
        df = df.dropna(subset=["voltage_v", "current_a", "temp_degc"])
        notes.append(f"dropped {n_nan} NaN rows")
    t = df["time_s"].to_numpy()
    back = np.flatnonzero(np.diff(t) < 0)
    if back.size:
        # Keep the longest run with no backward step and no gap above max_gap_s.
        # Only applied to files with restarts: HPPC rests are normally logged at 10 s.
        max_gap_s = 5.0
        breaks = np.flatnonzero((np.diff(t) < 0) | (np.diff(t) > max_gap_s)) + 1
        starts, ends = np.r_[0, breaks], np.r_[breaks, len(t)]
        j = int(np.argmax(ends - starts))
        a, b = starts[j], ends[j]
        notes.append(f"{back.size} time resets; kept longest contiguous run, raw t = {t[a]:.1f} to {t[b - 1]:.1f} s "
                     f"({b - a} of {len(t)} rows)")
        df = df.iloc[a:b]
    # Duplicate time stamps (about 5.7k rows over 177 files) would give zero-length
    # steps; keep the first row of each duplicate group. Applied to every file.
    df = df.drop_duplicates(subset="time_s", keep="first").reset_index(drop=True)
    return df, "; ".join(notes)


def c20_capacity(df: pd.DataFrame) -> dict:
    """Capacity from a C/20 test file (processed frame, i > 0 discharge).

    Discharge capacity: tester Ah counter change over the first sustained discharge
    segment. Charge capacity: change over the following sustained charge segment.
    """
    t, i, ah = df["time_s"].to_numpy(), df["current_a"].to_numpy(), df["ah"].to_numpy()
    seg = first_sustained_segment(t, i)
    if seg is None or seg[2] != 1:
        raise ValueError("C/20 file does not start with a discharge segment")
    a, b, _ = seg
    out = {"c20_discharge_ah": float(ah[b] - ah[a])}
    rest = first_sustained_segment(t[b + 1:], i[b + 1:])
    if rest is not None and rest[2] == -1:
        c, d = rest[0] + b + 1, rest[1] + b + 1
        out["c20_charge_ah"] = float(ah[c] - ah[d])
    return out


def add_soc_ref(df: pd.DataFrame, capacity_ah: float, ah_origin: float | None = None, soc0: float = 1.0) -> pd.DataFrame:
    """Reference SoC from the tester Ah counter (positive = discharged):
    soc_ref = soc0 - (Ah(t) - ah_origin) / capacity.

    ah_origin is the counter value at the start of the test. Pass the value from
    BEFORE any repair, so a file trimmed by repair_time keeps the charge removed
    before the kept segment (the counter runs through logging restarts).
    """
    df = df.copy()
    origin = df["ah"].iloc[0] if ah_origin is None else ah_origin
    df["soc_ref"] = soc0 - (df["ah"] - origin) / capacity_ah
    return df
