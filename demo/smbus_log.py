"""Log a fuel gauge's registers to CSV: an SBS smart battery over SMBus, or a TI BQ27426.

Read-only: only read transfers are issued (see socest/smbus_link.py and
socest/bq27426.py), so the gauge's settings cannot be changed.

    # emulated pack (no hardware), whole 25 degC UDDS cycle as fast as possible
    python demo/smbus_log.py --emulate 25degC_UDDS --speed 0 --identify

    # real pack on the Raspberry Pi's I2C bus 1 (GPIO 2 = SDA, GPIO 3 = SCL), 1 Hz, 30 min
    python demo/smbus_log.py --bus 1 --duration 1800 --scan 0x24-0x3f --identify

    # TI BQ27426 system-side gauge (one burst read per second, address 0x55)
    python demo/smbus_log.py --gauge bq27426 --bus 1 --duration 1800
    python demo/smbus_log.py --gauge bq27426 --emulate 25degC_UDDS

Output in demo/runs/smbus-<time>/: log.csv (decoded registers and raw words),
info.json (static registers, bus statistics) and, with --identify,
registers.md (what each scanned code tracks).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socest import register_id, sbs  # noqa: E402
from socest.smbus_link import LinuxI2C, SmartBatteryReader  # noqa: E402


def parse_codes(text: str) -> list[int]:
    out = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        a, _, b = part.partition("-")
        lo, hi = int(a, 16), int(b or a, 16)
        out += list(range(lo, hi + 1))
    return sorted(set(out) - set(sbs.COMMANDS))       # standard codes are read and decoded anyway


def flat(sample: dict) -> dict:
    out = {}
    for k, v in sample.items():
        if k == "BatteryStatus" and isinstance(v, dict):
            out["BatteryStatus_flags"] = "|".join(v["flags"])
            out["BatteryStatus_error"] = v["error"]
        elif k == "Flags" and isinstance(v, list):
            out["Flags"] = "|".join(v)
        else:
            out[k] = v
    return out


def open_source(args):
    if args.gauge == "bq27426":
        from socest.bq27426 import EmulatedBq27426

        if args.emulate:
            em = EmulatedBq27426(args.emulate, error_rate=args.error_rate)
            return em, em
        return LinuxI2C(args.bus), None
    if args.emulate:
        from socest.smart_battery_emulator import EmulatedSmartBattery

        em = EmulatedSmartBattery(args.emulate, pec=not args.no_pec, error_rate=args.error_rate, seed=0)
        return em, em
    return LinuxI2C(args.bus), None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bus", type=int, help="Linux I2C bus number (/dev/i2c-N)")
    src.add_argument("--emulate", metavar="CYCLE", help="emulated pack replaying a cycle from demo/replay/")
    ap.add_argument("--gauge", choices=["sbs", "bq27426"], default="sbs",
                    help="sbs: smart battery (SBS 1.1, address 0x0b); bq27426: TI BQ27426 (address 0x55)")
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=None, help="7-bit address (default per gauge)")
    ap.add_argument("--pec", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--no-pec", action="store_true", help="emulated pack without PEC support")
    ap.add_argument("--period", type=float, default=1.0, help="seconds between samples (real pack)")
    ap.add_argument("--speed", type=float, default=0.0, help="emulated samples per second (0 = as fast as possible)")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds of log time (0 = until Ctrl-C or end of cycle)")
    ap.add_argument("--scan", default="0x30-0x3f", help="extra codes to log as raw words, e.g. 0x24-0x3f,0x4f")
    ap.add_argument("--error-rate", type=float, default=0.0, help="emulated bus error probability per transfer")
    ap.add_argument("--identify", action="store_true", help="run register identification on the log at the end")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    transport, em = open_source(args)
    if args.gauge == "bq27426":
        from socest.bq27426 import ADDR, Bq27426Reader

        args.addr = args.addr or ADDR
        args.period = max(args.period, 1.0)            # the gauge allows at most 2 commands per second
        reader = Bq27426Reader(transport, addr=args.addr, min_interval_s=0.0 if em is not None else 0.5)
        scan, args.identify = [], False
    else:
        args.addr = args.addr or sbs.SMART_BATTERY_ADDR
        pec = {"auto": None, "on": True, "off": False}[args.pec]
        reader = SmartBatteryReader(transport, addr=args.addr, pec=pec, retry_delay_s=0.0 if em is not None else 0.02)
        scan = parse_codes(args.scan)
    info = reader.info()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or ROOT / "demo" / "runs" / f"smbus-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    if args.gauge == "bq27426":
        print(f"{info['device']} at {info['addr']}", flush=True)
    else:
        print(f"{info.get('ManufacturerName')} {info.get('DeviceName')} ({info.get('DeviceChemistry')}), "
              f"design {info.get('DesignCapacity')} Ah / {info.get('DesignVoltage')} V, "
              f"PEC {'on' if reader.pec else 'off'}; scanning {len(scan)} extra codes", flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *a: stop.update(flag=True))
    rows, t0, k = [], time.time(), 0
    while not stop["flag"]:
        if em is not None and em.done:
            break
        t_log = k * (1.0 if em is not None else args.period)
        if args.duration and t_log >= args.duration:
            break
        s = flat(reader.sample(scan=scan))
        s["t_s"] = t_log
        if em is not None:
            s["soc_ref_emulator"] = float(em.ref[em.k])
            em.advance()
            if args.speed > 0:
                time.sleep(max(0.0, t0 + (k + 1) / args.speed - time.time()))
        else:
            time.sleep(max(0.0, t0 + (k + 1) * args.period - time.time()))
        rows.append(s)
        k += 1
        if k % 600 == 0 or (em is None and k % 10 == 0):
            print(f"{k:6d} samples  V {s['Voltage']}  I {s['Current']}  T {s['Temperature']}  "
                  f"RSOC {s['RelativeStateOfCharge']} %  bus errors {reader.stats.failed}", flush=True)

    log = pd.DataFrame(rows)
    cols = ["t_s"] + [c for c in log if c != "t_s"]
    log[cols].to_csv(out / "log.csv", index=False)
    meta = {"source": f"emulated:{args.emulate}" if em is not None else f"i2c-{args.bus}", "addr": f"0x{args.addr:02x}",
            "info": info, "samples": len(log), "scan": [f"0x{c:02x}" for c in scan],
            "bus": {"ok": reader.stats.ok, "retried": reader.stats.retried, "failed": reader.stats.failed,
                    "pec_errors": reader.stats.pec_errors}}
    if args.gauge == "bq27426" and len(log):
        from socest.bq27426 import consistent

        meta["checks"] = {**consistent(log.dropna(subset=["Current"])), "sign_mismatch": reader.sign_mismatch,
                          "implausible": reader.implausible}
        print(f"checks: {meta['checks']}")
    (out / "info.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"\n{len(log)} samples -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}/log.csv; "
          f"bus: {meta['bus']}")
    if args.identify and len(log) > 10:
        table = register_id.identify(log, codes=scan)
        lines = [register_id.describe(r) for r in table.to_dict("records")]
        print("\nregister identification:\n  " + "\n  ".join(lines))
        (out / "registers.md").write_text("# Register identification\n\n" + table.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
