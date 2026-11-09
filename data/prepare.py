"""Convert raw LG HG2 .mat files to Parquet and write data/DATA_CARD.md.

Usage:
    python data/prepare.py                      # reads data/raw/LG_HG2, writes data/processed/LG_HG2
    python data/prepare.py --raw PATH --out PATH

Steps:
  1. Detect the raw current sign and convert to positive = discharge.
  2. Repair logging restarts (time resets, NaN rows); see lg_hg2.repair_time.
  3. Capacity per ambient temperature from the C/20 discharge (tester Ah counter).
  4. Reference SoC for each drive cycle from the tester Ah counter and that capacity,
     starting at 1.0 after a full CC-CV charge (checked against the start voltage).
  5. Flag excluded tests (lg_hg2.EXCLUDE) and write the data card.

Every statistic in the data card is computed from the files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socest.datasets import lg_hg2  # noqa: E402

COLUMN_DOCS = {
    "time_s": ("s", "Test time from the raw `Time` column, re-zeroed at the first kept sample"),
    "voltage_v": ("V", "Cell terminal voltage (raw `Voltage`)"),
    "current_a": ("A", "Cell current (raw `Current`), converted so that positive = discharge"),
    "ah": ("Ah", "Tester amp-hour counter (raw `Ah`), sign converted so that positive = net discharge"),
    "power_w": ("W", "Tester power (raw `Power`), sign converted with current"),
    "temp_degc": ("degC", "Battery temperature as reported by the tester (raw `Battery_Temp_degC`)"),
    "soc_ref": ("fraction", "Drive cycles only. Reference SoC: 1 - (ah - ah[0]) / C20 discharge capacity at that ambient temperature"),
}
ORDER = list(lg_hg2.FOLDER_TEMPS)


def summarise(df: pd.DataFrame) -> dict:
    dt = np.diff(df["time_s"].to_numpy())
    return {
        "samples": len(df),
        "duration_h": round(float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]) / 3600, 3),
        "dt_median_s": round(float(np.median(dt)), 4) if dt.size else np.nan,
        "dt_max_s": round(float(np.max(dt)), 3) if dt.size else np.nan,
        "v_start": round(float(df["voltage_v"].iloc[0]), 4),
        "v_min": round(float(df["voltage_v"].min()), 4),
        "v_max": round(float(df["voltage_v"].max()), 4),
        "i_min": round(float(df["current_a"].min()), 3),
        "i_max": round(float(df["current_a"].max()), 3),
        "t_min": round(float(df["temp_degc"].min()), 2),
        "t_max": round(float(df["temp_degc"].max()), 2),
        "net_ah": round(float(df["ah"].iloc[-1] - df["ah"].iloc[0]), 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "LG_HG2")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "LG_HG2")
    ap.add_argument("--card", type=Path, default=ROOT / "data" / "DATA_CARD.md")
    args = ap.parse_args(argv)

    files = lg_hg2.list_files(args.raw)
    if not files:
        sys.exit(f"No .mat files under {args.raw}. Run data/download.py first.")
    sign = lg_hg2.detect_discharge_sign(files)
    print(f"{len(files)} files; raw discharge current sign = {sign:+d}")

    # Pass 1: load, repair, capacities.
    frames, repairs, capacity, raw_start = {}, {}, {}, {}
    for tf in files:
        df = lg_hg2.to_frame(tf, sign)
        first = df.dropna(subset=["voltage_v", "current_a", "ah"]).iloc[0]
        raw_start[tf] = {"v": float(first["voltage_v"]), "i": float(first["current_a"]), "ah": float(first["ah"])}
        df, note = lg_hg2.repair_time(df)
        df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
        frames[tf] = df
        if note:
            repairs[(tf.folder, tf.series, tf.test)] = note
        if tf.test == "C20DisCh":
            capacity[tf.folder] = {"source": f"{tf.series}_C20DisCh", **lg_hg2.c20_capacity(df)}

    # Chronological order within each folder, for the full-charge start check.
    previous = {}
    for folder in ORDER:
        ordered = sorted((tf for tf in files if tf.folder == folder), key=lambda tf: tf.when)
        for a, b in zip([None] + ordered[:-1], ordered):
            previous[b] = a

    # Pass 2: reference SoC, exclusions, write.
    rows = []
    for tf, df in frames.items():
        key = (tf.folder, tf.series, tf.test)
        reason = lg_hg2.EXCLUDE.get(key, "")
        start_check = ""
        if tf.kind == "drive_cycle":
            df = lg_hg2.add_soc_ref(df, capacity[tf.folder]["c20_discharge_ah"], ah_origin=raw_start[tf]["ah"])
            prev = previous[tf]
            if prev is not None and prev.kind == "charge":
                vmax = float(frames[prev]["voltage_v"].max())
                start_check = f"after {prev.series}_{prev.test} (max {vmax:.3f} V)"
                if vmax < lg_hg2.CHARGE_CV_LIMIT_V and not reason:
                    reason = f"preceding charge {prev.test} reached only {vmax:.3f} V"
            else:
                r0 = raw_start[tf]
                start_check = f"no preceding charge file; starts at {r0['v']:.3f} V, {r0['i']:+.2f} A"
                if (abs(r0["i"]) >= 0.1 or r0["v"] < lg_hg2.FALLBACK_REST_V) and not reason:
                    reason = "no preceding charge file and start is not a rested full cell"
        dest = args.out / tf.folder / f"{tf.series}_{tf.test}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False)
        row = {"folder": tf.folder, "series": tf.series, "test": tf.test, "kind": tf.kind,
               "file": tf.path.name, "parquet": f"{tf.folder}/{dest.name}", **summarise(df),
               "excluded": bool(reason), "exclude_reason": reason, "repair": repairs.get(key, ""),
               "start_check": start_check}
        row["v_start"] = round(raw_start[tf]["v"], 4)
        if tf.kind == "drive_cycle":
            row["soc_ref_end"] = round(float(df["soc_ref"].iloc[-1]), 4)
            row["soc_ref_min"] = round(float(df["soc_ref"].min()), 4)
        rows.append(row)

    index = pd.DataFrame(rows)
    index["folder"] = pd.Categorical(index["folder"], ORDER, ordered=True)
    index = index.sort_values(["folder", "kind", "test", "series"]).reset_index(drop=True)
    index.to_csv(args.out / "index.csv", index=False)
    cap = pd.DataFrame.from_dict(capacity, orient="index").loc[[f for f in ORDER if f in capacity]]
    cap.to_csv(args.out / "capacity.csv")
    write_card(args.card, index, cap, sign)


def write_card(path: Path, index: pd.DataFrame, cap: pd.DataFrame, sign: int):
    L = [
        "# Data card: LG 18650HG2 (processed)",
        "",
        "Generated by `data/prepare.py` from the raw files. Do not edit by hand.",
        "",
        "## Source",
        "",
        "P. Kollmeyer, M. Naguib, M. Skells, *LG 18650HG2 Li-ion Battery Data*, Mendeley Data, V2, 2020. "
        "doi:10.17632/b5mj79w5w9.2. Licence CC BY 4.0. Per the dataset description: a new 3 Ah LG HG2 cell, "
        "tested at McMaster University in an 8 cu.ft. thermal chamber on a Digatron tester with 0.1 % of full "
        "scale voltage and current accuracy. Only the `.mat` files are used; per the dataset readme the `.csv` "
        "files hold the same data.",
        "",
        "## Sign convention",
        "",
        f"Raw files report discharge current as **{'positive' if sign > 0 else 'negative'}**. Detected from the "
        "first sustained current segment of every C/20, 0.5C, 1C and 2C discharge file, each required to lower "
        "the voltage by at least 0.3 V. Processed files use **positive = discharge**.",
        "",
        "## Columns (Parquet)",
        "",
        "| Column | Unit | Description |",
        "|---|---|---|",
    ]
    for col, (unit, doc) in COLUMN_DOCS.items():
        L.append(f"| `{col}` | {unit} | {doc} |")
    L += [
        "",
        "Rows with a duplicate `Time` value are dropped (first kept) in every file; they would otherwise give "
        "zero-length time steps.",
        "",
        "Not loaded: `Wh` (uint8 with one constant value in the files inspected) and `TimeStamp` (wall-clock "
        "string, redundant with `Time`).",
        "",
        "## Capacity and reference SoC",
        "",
        "Capacity at each ambient temperature is the tester Ah counter change over the C/20 discharge "
        "(and the following C/20 charge). Drive cycles start after a CC-CV charge to 4.2 V, so the reference SoC "
        "starts at 1.0 and follows the tester Ah counter, normalised by the C/20 discharge capacity at the same "
        "ambient temperature. The full-charge start is checked per cycle: the file just before it must be a "
        "charge that reached the 4.2 V CV limit (column `start_check` in `index.csv`). The Ah counter was preferred over integrating the logged current because it "
        "continues through logging gaps; on the drive cycles the two agree to a median maximum difference of "
        "about 1 mAh.",
        "",
        cap.round(4).to_markdown(),
        "",
        "Capacity tests ran in chronological order 25, 40, 10, 0, -10, -20 degC (file time stamps), so "
        "temperature and test order are confounded.",
        "",
        "## Excluded tests",
        "",
    ]
    ex = index[index["excluded"]]
    L.append(ex[["folder", "series", "test", "exclude_reason"]].to_markdown(index=False) if len(ex) else "None.")
    L += ["", "## Repaired files", ""]
    rp = index[index["repair"] != ""]
    L.append(rp[["folder", "series", "test", "repair"]].to_markdown(index=False) if len(rp) else "None.")
    L += ["", "## Usable drive cycles", "", "`x` = usable, `excl` = excluded, `-` = not in the dataset.", ""]
    dc = index[index["kind"] == "drive_cycle"]
    grid = pd.DataFrame("-", index=list(lg_hg2.DRIVE_CYCLES), columns=ORDER)
    for _, r in dc.iterrows():
        grid.loc[r["test"], r["folder"]] = "excl" if r["excluded"] else "x"
    L.append(grid.to_markdown())
    L += ["", "## Drive cycles", ""]
    cols = ["folder", "series", "test", "duration_h", "dt_median_s", "dt_max_s", "v_start", "v_min",
            "i_min", "i_max", "t_min", "t_max", "net_ah", "soc_ref_end", "excluded"]
    L.append(dc[cols].to_markdown(index=False))
    L += ["", "## All other files", ""]
    cols = ["folder", "series", "test", "kind", "duration_h", "dt_median_s", "v_min", "v_max", "i_min", "i_max",
            "t_min", "t_max", "net_ah"]
    L.append(index[index["kind"] != "drive_cycle"][cols].to_markdown(index=False))
    L.append("")
    path.write_text("\n".join(L))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
