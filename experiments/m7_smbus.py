"""M7: smart-battery (SMBus) input path, end to end on an emulated pack.

For each recorded test cycle in demo/replay/:
  1. serve it from an emulated SBS 1.1 smart battery and read it back over the
     full SMBus stack (PEC on, 1 % emulated bus errors, 0.5 % corrupted replies),
     logging the standard registers and the vendor codes 0x30-0x3F
  2. run the estimators on the SBS-resolution data (1 mV, 1 mA, 0.1 K) and on
     the original recording, from the same correct initial SoC
  3. identify the vendor registers from the log without being told their meaning,
     and score the result against what the emulator actually put there

The emulated gauge's RelativeStateOfCharge() (OCV start + coulomb counting with
the measured capacity) is reported for completeness; it is not a model of a
commercial fuel gauge.

Writes results/tables/m7_smbus.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socest import register_id  # noqa: E402
from socest.estimators import EKFEstimator, selected_networks  # noqa: E402
from socest.onnx_estimator import OnnxNetwork  # noqa: E402
from socest.replay import load_replay, replay_names  # noqa: E402
from socest.smart_battery_emulator import EmulatedSmartBattery  # noqa: E402
from socest.smbus_link import SmartBatteryReader  # noqa: E402
from socest.streaming import StreamingCoulomb, StreamingEKF  # noqa: E402

SCAN = list(range(0x30, 0x40))
# What the emulator writes at each code (see socest.smart_battery_emulator.VENDOR_TRUTH): quantity, scale per count.
EXPECTED = {0x30: ("Temperature", 0.1), 0x31: ("Current", 0.01), 0x32: ("discharged_mah", 1.0),
            0x33: ("constant", None), 0x3C: ("constant", None), 0x3D: ("constant", None), 0x3E: ("constant", None),
            0x3F: ("Voltage", 0.001), **{c: ("no answer", None) for c in range(0x34, 0x3C)}}
NAMES = {"ekf": "EKF, 3RC (calibrated)", "lstm": "LSTM", "transformer": "Transformer", "cc": "Coulomb counting"}


def log_cycle(cycle: str, seed: int = 0):
    em = EmulatedSmartBattery(cycle, pec=True, error_rate=0.01, corrupt_rate=0.005, seed=seed)
    reader = SmartBatteryReader(em, retry_delay_s=0.0)
    rows = []
    while not em.done:
        s = reader.sample(scan=SCAN)
        s["t_s"] = float(em.t[em.k])
        rows.append(s)
        em.advance()
    return pd.DataFrame(rows), reader.stats


def estimate(v, i, T, soc0, nets, ekf_est):
    out = {}
    ekf = StreamingEKF(ekf_est, temp0=float(T[0]), soc0=soc0)
    out["ekf"] = np.array([ekf.step(a, b, c) for a, b, c in zip(v, i, T)])
    for key in ("lstm", "transformer"):
        out[key] = np.asarray(OnnxNetwork(nets[key])(v, i, T))
    cc = StreamingCoulomb(ekf_est.cap_of(float(T[0])), soc0=soc0)
    out["cc"] = np.array([cc.step(a, b, c) for a, b, c in zip(v, i, T)])
    return out


def rmse_pts(x, ref):
    return float(100 * np.sqrt(np.mean((np.asarray(x) - ref) ** 2)))


def main():
    nets = selected_networks()
    ekf_est = EKFEstimator.tuned(None)
    acc_rows, bus_rows, id_rows, res_rows = [], [], [], []
    for cycle in replay_names():
        u = load_replay(cycle)
        ref = u.soc_ref.to_numpy(float)
        log, stats = log_cycle(cycle)
        assert len(log) == len(u)
        raw = estimate(u.voltage_v.to_numpy(float), u.current_a.to_numpy(float), u.temp_degc.to_numpy(float),
                       float(ref[0]), nets, ekf_est)
        bus = estimate(log.Voltage.to_numpy(float), log.Current.to_numpy(float), log.Temperature.to_numpy(float),
                       float(ref[0]), nets, ekf_est)
        for key in NAMES:
            a, b = rmse_pts(raw[key], ref), rmse_pts(bus[key], ref)
            acc_rows.append({"Cycle": cycle, "Estimator": NAMES[key], "RMSE, recorded data": round(a, 3),
                             "RMSE, read over SMBus": round(b, 3), "Difference": round(b - a, 3)})
        acc_rows.append({"Cycle": cycle, "Estimator": "Emulated gauge, RelativeStateOfCharge()",
                         "RMSE, recorded data": None,
                         "RMSE, read over SMBus": round(rmse_pts(log.RelativeStateOfCharge.to_numpy(float) / 100, ref), 3),
                         "Difference": None})
        res_rows.append({"Cycle": cycle,
                         "Voltage, max abs (mV)": round(1000 * np.max(np.abs(log.Voltage - u.voltage_v)), 3),
                         "Current, max abs (mA)": round(1000 * np.max(np.abs(log.Current - u.current_a)), 3),
                         "Temperature, max abs (K)": round(float(np.max(np.abs(log.Temperature - u.temp_degc))), 3)})
        bus_rows.append({"Cycle": cycle, "Samples": len(log), "Reads OK": stats.ok, "Retried": stats.retried,
                         "PEC mismatches caught": stats.pec_errors, "Failed after retries": stats.failed})
        table = register_id.identify(log, codes=SCAN)
        for r in table.to_dict("records"):
            code = int(r["code"], 16)
            what, scale = EXPECTED[code]
            if what in ("constant", "no answer"):
                correct = r["verdict"] == what
            else:   # right quantity, right scale and sign
                correct = r["verdict"] == "match" and r["tracks"] == what and abs(r["scale"] - scale) < 1e-9 * scale
            id_rows.append({"Cycle": cycle, "Code": r["code"],
                            "Emulator put there": what + (f", {scale:g} per count" if scale else ""),
                            "Identified as": register_id.describe(r).split(": ", 1)[1], "Correct": bool(correct)})

    acc, bus, ids, res = (pd.DataFrame(x) for x in (acc_rows, bus_rows, id_rows, res_rows))
    n_ok, n = int(ids.Correct.sum()), len(ids)
    shown = ids[ids["Emulator put there"] != "no answer"]
    worst = acc.dropna().Difference.abs().max()
    md = [
        "# M7: smart-battery (SMBus) input path on an emulated pack",
        "",
        "Each test cycle is served by an emulated SBS 1.1 smart battery (`src/socest/smart_battery_emulator.py`) and read "
        "back over the full SMBus stack (`src/socest/smbus_link.py`): byte-level transfers, PEC on, 1 % of transfers "
        "fail like a NACK and 0.5 % of replies have a flipped bit. The emulated pack is one LG HG2 cell; its gauge is a "
        "simple OCV start plus coulomb counting, not a model of a commercial fuel gauge.",
        "",
        "## Bus",
        "",
        bus.to_markdown(index=False),
        "",
        "## SBS resolution vs the recording",
        "",
        "Largest difference between the decoded SBS registers (1 mV, 1 mA, 0.1 K) and the recorded values:",
        "",
        res.to_markdown(index=False),
        "",
        "## SoC accuracy on data read over SMBus (percentage points, correct initial SoC)",
        "",
        acc.to_markdown(index=False),
        "",
        f"Largest change in RMSE from reading the data over SMBus: {worst:.3f} points.",
        "",
        "## Register identification (vendor codes 0x30-0x3F)",
        "",
        f"Correct (right quantity, scale and sign, or right constant / no-answer verdict) on {n_ok} of {n} code-cycle pairs; "
        "codes that never answered (0x34-0x3B) are left out of the table.",
        "",
        shown.to_markdown(index=False),
        "",
    ]
    out = ROOT / "results" / "tables" / "m7_smbus.md"
    out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
