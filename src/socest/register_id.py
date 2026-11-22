"""Identify undocumented smart-battery registers from logged data.

Gauges put useful data (cell voltages, temperatures, counters) at
manufacturer-specific SMBus codes, and the layout differs between gauge
families: for example CellVoltage1() is 0x3F on the TI BQ4050 but CellVoltage4()
is 0x3F on the BQ40Z80. Instead of trusting a vendor table, this module looks
at a log of raw words and tests, for every unknown code, which known quantity it
tracks.

For each unknown code it tries the raw word as unsigned and as signed 16-bit,
fits  known = a * word + b  by least squares against every reference signal,
and keeps the best fit:

    match        the fit residual is within one count of either side's resolution
    likely       R^2 >= 0.98, but not within one count
    constant     the word never changed during the log
    unexplained  nothing fits

Reference signals: the decoded standard registers (Voltage, Current,
AverageCurrent, Temperature, RelativeStateOfCharge, RemainingCapacity) and two
derived ones, the integrals of discharge current and of net current (mAh).
A match means the register moves with that quantity during this log, with the
reported scale and offset. A log that exercises more conditions (charge,
discharge, temperature changes) separates quantities that happen to move
together in a short log.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REFERENCES = {   # column: (unit, resolution of the decoded value)
    "Voltage": ("V", 1e-3),
    "Current": ("A", 1e-3),            # decoded with positive = discharge
    "AverageCurrent": ("A", 1e-3),
    "Temperature": ("degC", 0.1),
    "RelativeStateOfCharge": ("%", 1.0),
    "RemainingCapacity": ("Ah", 1e-3),
    "discharged_mah": ("mAh", 1.0),
    "net_discharged_mah": ("mAh", 1.0),
}


def add_derived(log: pd.DataFrame, dt: float | None = None) -> pd.DataFrame:
    """Integrals of current over the log (mAh), from the decoded Current() column."""
    log = log.copy()
    t = log["t_s"].to_numpy(float) if "t_s" in log else np.arange(len(log), dtype=float)
    step = np.diff(t, prepend=t[0]) if dt is None else np.full(len(t), dt)
    i = log["Current"].astype(float).fillna(0.0).to_numpy()
    i_prev = np.concatenate([[0.0], i[:-1]])            # current during the step that ends at this sample
    log["discharged_mah"] = np.cumsum(np.maximum(i_prev, 0.0) * step) / 3.6
    log["net_discharged_mah"] = np.cumsum(i_prev * step) / 3.6
    return log


def round_scales(x: float, spread: float = 0.25) -> list[float]:
    """Round scales (+-{1, 2, 2.5, 5} x 10^k) within `spread` of x, closest first."""
    if x == 0 or not np.isfinite(x):
        return []
    e = np.floor(np.log10(abs(x)))
    c = [np.sign(x) * m * 10 ** (e + d) for d in (-1, 0, 1) for m in (1, 2, 2.5, 5)]
    c = [float(v) for v in c if abs(v - x) <= spread * abs(x)]
    return sorted(c, key=lambda v: abs(v - x))


def _fit(x, y):
    A = np.column_stack([x, np.ones_like(x)])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - (a * x + b)
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(res**2) / ss if ss > 0 else 0.0
    return float(a), float(b), float(r2), float(np.sqrt(np.mean(res**2)))


def identify(log: pd.DataFrame, codes=None, references=REFERENCES, min_answer: float = 0.5) -> pd.DataFrame:
    """One row per unknown code: verdict, best reference, scale, offset, R^2, residual, runner-up."""
    if "discharged_mah" not in log:
        log = add_derived(log)
    codes = codes if codes is not None else sorted(int(c[6:], 16) for c in log if c.startswith("raw_0x"))
    refs = {k: log[k].astype(float).to_numpy() for k in references if k in log}
    rows = []
    for code in codes:
        col = f"raw_0x{code:02x}"
        if col not in log:
            continue
        raw = pd.to_numeric(log[col], errors="coerce").to_numpy(float)
        ok = np.isfinite(raw)
        row = {"code": f"0x{code:02x}", "answered": round(float(ok.mean()), 3)}
        if ok.mean() < min_answer:
            rows.append({**row, "verdict": "no answer"})
            continue
        w = raw[ok].astype(np.int64)
        if np.all(w == w[0]):
            rows.append({**row, "verdict": "constant", "value": f"0x{w[0]:04x} ({w[0]})"})
            continue
        cands = []
        interps = {"u16": w.astype(float)}
        if np.any(w >= 0x8000):
            interps["i16"] = np.where(w >= 0x8000, w - 0x10000, w).astype(float)
        for kind, x in interps.items():
            for name, y_all in refs.items():
                y = y_all[ok]
                good = np.isfinite(y)
                if good.sum() < 10 or np.std(y[good]) == 0:
                    continue
                a, b, r2, rms = _fit(x[good], y[good])
                tol = max(references[name][1], abs(a))      # one count of either side
                # Prefer a round scale when it fits as well. Quantisation noise in a short or
                # narrow log biases the least-squares slope low, so look up to 25 % away.
                for an in round_scales(a):
                    bn = float(np.mean(y[good] - an * x[good]))
                    rms_n = float(np.sqrt(np.mean((y[good] - an * x[good] - bn) ** 2)))
                    if rms_n <= max(tol, 1.05 * rms):
                        a, b, rms = an, bn, rms_n
                        r2 = float(1 - rms**2 / np.var(y[good]))
                        tol = max(references[name][1], abs(a))
                        break
                cands.append({"kind": kind, "ref": name, "a": a, "b": b, "r2": r2, "rms": rms, "tol": tol,
                              "within": rms <= tol})
        if not cands:
            rows.append({**row, "verdict": "unexplained"})
            continue
        # Best fit first; between equally good fits (unsigned vs signed reading of the same
        # word) the one with the smaller offset is the natural reading.
        cands.sort(key=lambda c: (not c["within"], -round(c["r2"], 6), abs(c["b"])))
        best = cands[0]
        second = next((c for c in cands[1:] if c["ref"] != best["ref"]), None)
        verdict = "match" if best["within"] else "likely" if best["r2"] >= 0.98 else "unexplained"
        unit = references[best["ref"]][0]
        rows.append({**row, "verdict": verdict, "type": best["kind"], "tracks": best["ref"],
                     "scale": best["a"], "offset": round(best["b"], 4), "unit": unit,
                     "r2": round(best["r2"], 6), "rms_resid": float(f"{best['rms']:.3g}"),
                     "runner_up": f"{second['ref']} (R^2 {second['r2']:.3f})" if second else ""})
    return pd.DataFrame(rows)


def describe(row: dict) -> str:
    """One readable line for a row of identify()."""
    tracks = row.get("tracks")
    if not isinstance(tracks, str):
        extra = f" {row['value']}" if row["verdict"] == "constant" else ""
        return f"{row['code']}: {row['verdict']}{extra}"
    if row["verdict"] == "unexplained":
        return f"{row['code']}: unexplained (best {row['tracks']}, R^2 {row['r2']})"
    off = f", offset {row['offset']:g}" if abs(row["offset"]) > 1e-9 else ""
    sign = " (current positive = discharge)" if "Current" in tracks else ""
    return (f"{row['code']}: {row['verdict']} {tracks}, {row['type']}, {row['scale']:g} {row['unit']} per count"
            f"{off}{sign}; R^2 {row['r2']}, residual {row['rms_resid']:g}")
