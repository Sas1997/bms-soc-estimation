"""Edge node of the cloud-connected BMS demo: estimate SoC on the device, stream to the cloud.

Takes one sample at a time from a measurement source and runs four estimators
on every sample. Sources (--source):

    replay          a test cycle from demo/replay/, as if it came from a live cell (default)
    smbus-emulated  the same cycle served by an emulated smart battery and read over the
                    full SMBus stack (bytes, PEC, retries, SBS decoding); adds the pack
                    fuel gauge's RelativeStateOfCharge() as a fifth series
    smbus           a real smart battery on the Pi's I2C bus (read-only); the pack's fuel
                    gauge is the reference. The EKF and the networks model an LG HG2 cell,
                    so their numbers mean something only for a pack built from that cell.
    bq27426         a TI BQ27426 system-side Impedance Track gauge on the Pi's I2C bus
                    (read-only, one burst read per second); its StateOfCharge() is the
                    reference. Same caveat about the cell.
    bq27426-emulated  the BQ27426 register interface served from the recorded cycle

Estimators:

    EKF            selected cell model (3RC, calibrated) with the tuned noise settings
    LSTM           ONNX Runtime, exported from the selected checkpoint
    Transformer    ONNX Runtime, exported from the selected checkpoint
    Coulomb        open-loop current integration

The EKF and the coulomb counter start --soc-error points below the true SoC, the
way a BMS that has lost its state would. The networks have no initial state.

Every second (wall clock) it publishes the new samples and estimates over MQTT
(TLS, username and password) to the broker named in demo/config.toml, and every
30 s a retained, downsampled history of the run so far, so a dashboard opened
mid-run (docs/index.html) shows the whole run. Everything is also logged to
demo/runs/<run id>/ on the device.

    python demo/edge.py --offline                 # no network: check the install, log locally
    python demo/edge.py                           # stream to the broker in demo/config.toml
    python demo/edge.py --cycle n20degC_UDDS --speed 1
    python demo/edge.py --source smbus-emulated --offline
    python demo/edge.py --source smbus --bus 1 --series 1 --parallel 1
    python demo/edge.py --source bq27426 --bus 1

Topics, under the prefix set in the config (default bms-soc-estimation/demo):
    online     retained, "1" while running, "0" otherwise (also the MQTT last will)
    meta       retained, run description
    telemetry  new samples since the last message
    history    retained, downsampled run so far
    summary    retained, final metrics when the run ends
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socest import sbs  # noqa: E402
from socest.estimators import EKFEstimator, selected_networks  # noqa: E402
from socest.onnx_estimator import OnnxNetwork  # noqa: E402
from socest.replay import load_replay, replay_names  # noqa: E402
from socest.smbus_link import LinuxI2C, SmartBatteryReader, cell_inputs  # noqa: E402
from socest.streaming import StreamingCoulomb, StreamingEKF  # noqa: E402

KEYS = ["ekf", "lstm", "transformer", "cc"]
LABELS = {"ekf": "EKF, 3RC (calibrated)", "lstm": "LSTM", "transformer": "Transformer", "cc": "Coulomb counting",
          "gauge": "Pack fuel gauge, RelativeStateOfCharge()"}
SBS_READ = ["Voltage", "Current", "Temperature", "RelativeStateOfCharge", "BatteryStatus"]
HISTORY_POINTS = 1500          # max points in the retained history message
HISTORY_EVERY_S = 30.0
PUBLISH_EVERY_S = 1.0


# ---------------------------------------------------------------- helpers
def cpu_temp_c():
    """SoC temperature of a Raspberry Pi (or any Linux board exposing thermal_zone0), else None."""
    try:
        return round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0, 1)
    except (OSError, ValueError):
        return None


def device_name():
    try:
        return Path("/proc/device-tree/model").read_text().strip("\x00").strip()
    except OSError:
        return f"{platform.system()} {platform.machine()}"


def _cmd(*args, timeout=5):
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def power_status():
    """Raspberry Pi firmware power report (None elsewhere): throttling flags and the 5 V input voltage."""
    thr = _cmd("vcgencmd", "get_throttled")
    volt = _cmd("vcgencmd", "pmic_read_adc", "EXT5V_V")
    v = None
    if volt and "=" in volt:
        try:
            v = round(float(volt.split("=")[-1].rstrip("V")), 3)
        except ValueError:
            pass
    return {"throttled": thr.split("=")[-1] if thr else None, "ext5v_v": v}


def hardware_info():
    """What this process actually ran on, read from the system rather than from the config file."""
    info = {"model": device_name(), "machine": platform.machine(), "kernel": platform.release(),
            "cpus": os.cpu_count(), "python": platform.python_version()}
    try:
        rel = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
        info["os"] = rel.get("PRETTY_NAME", "").strip('"') or None
    except OSError:
        info["os"] = f"{platform.system()} {platform.version()}"
    try:
        kb = int(next(l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal")).split()[1])
        info["memory_gb"] = round(kb / 1024**2, 1)
    except (OSError, StopIteration, ValueError):
        info["memory_gb"] = None
    commit = _cmd("git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD")
    dirty = _cmd("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no")
    info["code_commit"] = commit + ("+modified" if dirty else "") if commit else None
    info.update(power_status())
    return info


def r(x, d):
    """Round for JSON; None for NaN."""
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), d)


def load_config(path: Path):
    import tomllib

    if not path.exists():
        sys.exit(f"{path} not found. Copy demo/config.example.toml to demo/config.toml and fill it in, "
                 "or run with --offline.")
    cfg = tomllib.loads(path.read_text())
    b = cfg.get("broker", {})
    for k in ("host", "username", "password"):
        if not b.get(k) or "your-" in str(b.get(k)):
            sys.exit(f"demo/config.toml: [broker] {k} is not filled in.")
    return cfg


class Publisher:
    """Thin wrapper around paho-mqtt; does nothing when offline."""

    def __init__(self, cfg, topic: str, offline: bool):
        self.topic, self.client = topic, None
        if offline:
            return
        import paho.mqtt.client as mqtt

        b = cfg["broker"]
        cid = f"bms-edge-{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid, protocol=mqtt.MQTTv311)
        c.username_pw_set(b["username"], b["password"])
        if b.get("tls", True):
            c.tls_set()                                   # system CA certificates
        c.will_set(f"{topic}/online", "0", qos=1, retain=True)
        self.status = None

        def on_connect(client, userdata, flags, reason_code, properties):
            self.status = reason_code

        c.on_connect = on_connect
        c.reconnect_delay_set(1, 30)
        port = int(b.get("port", 8883))
        print(f"connecting to {b['host']}:{port} as {b['username']} ...", flush=True)
        c.connect(b["host"], port, keepalive=30)
        c.loop_start()
        t0 = time.time()
        while self.status is None and time.time() - t0 < 15:
            time.sleep(0.05)
        if self.status is None or self.status.is_failure:
            c.loop_stop()
            sys.exit(f"MQTT connection failed: {self.status or 'no answer within 15 s'}. "
                     "Check host, port, username and password in demo/config.toml.")
        print("connected", flush=True)
        self.client = c

    def send(self, sub, payload, retain=False, wait=False):
        if self.client is None:
            return
        data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
        info = self.client.publish(f"{self.topic}/{sub}", data, qos=1, retain=retain)
        if wait:
            info.wait_for_publish(10)

    def close(self):
        if self.client is not None:
            self.send("online", "0", retain=True, wait=True)
            self.client.disconnect()
            self.client.loop_stop()


# ---------------------------------------------------------------- sources
class ReplaySource:
    kind, gauge, ref_label = "replay", False, "laboratory coulomb counting"

    def __init__(self, cycle: str):
        u = load_replay(cycle)
        self.name = cycle
        self.cols = [u[c].to_numpy(float) for c in ("time_s", "voltage_v", "current_a", "temp_degc", "soc_ref")]
        self.n = len(u)

    def get(self, k):
        t, v, i, T, ref = (c[k] for c in self.cols)
        return t, v, i, T, ref, None


class SMBusSource:
    """Samples from a fuel gauge (SBS smart battery or BQ27426), real or emulated.

    Holds the last good value on a failed read. With a real gauge (no emulator),
    the gauge's own SoC is the reference.
    """

    gauge = True

    def __init__(self, reader, emulator=None, n_series=1, n_parallel=1, cell_voltage_cmd=None,
                 n=86400, period=1.0, family="smbus"):
        self.r, self.em, self.ns, self.np, self.cvc = reader, emulator, n_series, n_parallel, cell_voltage_cmd
        self.live = emulator is None
        self.kind = family + ("-emulated" if emulator is not None else "")
        self.name = emulator.cycle if emulator is not None else "live pack"
        self.n = emulator.n if emulator is not None else n
        self.period, self.t0 = period, None
        self.ref_label = ("laboratory coulomb counting (recorded cycle)" if emulator is not None
                          else "the pack's own fuel gauge")
        self.last, self.stale, self.last_sample = None, 0, None
        if family == "bq27426" and emulator is None:
            self.ref_label = "the BQ27426's own StateOfCharge()"
            self.name = "BQ27426 cell"
        self.read_ms = []

    def get(self, k):
        s = self.r.sample(names=SBS_READ)
        if s.get("Voltage") is not None:
            self.last_sample = s
        self.read_ms.append(s["t_read_ms"])
        cv = None
        if self.cvc is not None:
            w = self.r.try_word(self.cvc)
            cv = w / 1000.0 if w is not None else None
        v, i, T = cell_inputs(s, self.ns, self.np, cv)
        g = s["RelativeStateOfCharge"] / 100.0 if s["RelativeStateOfCharge"] is not None else None
        cur = (v, i, T, g)
        if any(x is None for x in cur):
            if self.last is None:
                raise OSError("first SMBus sample incomplete; check wiring, address and pull-ups")
            self.stale += 1
            cur = tuple(c if c is not None else p for c, p in zip(cur, self.last))
        self.last = cur
        v, i, T, g = cur
        if self.em is not None:
            ref, t = float(self.em.ref[self.em.k]), float(self.em.t[self.em.k])
            self.em.advance()
        else:
            self.t0 = time.time() if self.t0 is None else self.t0
            ref, t = g, round(time.time() - self.t0, 1)
        return t, v, i, T, ref, g


# ---------------------------------------------------------------- run
class Run:
    def __init__(self, source, soc_error_pts: float, threads: int, n: int | None = None, capacity_ah=None):
        self.src, self.cycle = source, source.name
        self.n = n or source.n
        self.keys = KEYS + (["gauge"] if source.gauge and not getattr(source, "live", False) else [])
        self.labels = dict(LABELS)
        if source.kind == "smbus-emulated":
            self.labels["gauge"] = "Pack fuel gauge (emulated: OCV start, coulomb counting)"
        elif source.kind == "bq27426-emulated":
            self.labels["gauge"] = "BQ27426 interface (emulated gauge: OCV start, coulomb counting)"
        self.t, self.v, self.i, self.T, self.ref, self.gauge = (np.full(self.n, np.nan) for _ in range(6))
        self.k = 0
        self._take()                                      # first sample: initial SoC and temperature
        soc0 = float(self.ref[0]) - soc_error_pts / 100.0
        T0 = float(self.T[0])
        nets = selected_networks()
        ekf = EKFEstimator.tuned(None, label=LABELS["ekf"])
        self.models = {"ekf": ekf.variant, "lstm": nets["lstm"], "transformer": nets["transformer"]}
        self.est = {
            "ekf": StreamingEKF(ekf, temp0=T0, soc0=soc0),
            "lstm": OnnxNetwork(nets["lstm"], threads=threads),
            "transformer": OnnxNetwork(nets["transformer"], threads=threads),
            "cc": StreamingCoulomb(capacity_ah or ekf.cap_of(T0), soc0=soc0),
        }
        self.out = {k: np.full(self.n, np.nan) for k in self.keys}
        self.ekf_sd = np.full(self.n, np.nan)
        self.ms = {k: np.full(self.n, np.nan) for k in KEYS}

    def _take(self):
        k = self.k
        self.t[k], self.v[k], self.i[k], self.T[k], self.ref[k], g = self.src.get(k)
        self.gauge[k] = np.nan if g is None else g
        self._pending = True

    def step(self):
        k = self.k
        if not getattr(self, "_pending", False):
            self._take()
        self._pending = False
        v, i, T = self.v[k], self.i[k], self.T[k]
        for key, est in self.est.items():
            t0 = time.perf_counter_ns()
            self.out[key][k] = est.step(v, i, T)
            self.ms[key][k] = (time.perf_counter_ns() - t0) / 1e6
        if "gauge" in self.out:
            self.out["gauge"][k] = self.gauge[k]
        self.ekf_sd[k] = self.est["ekf"].soc_std
        self.k += 1

    def block(self, a, b, every=1):
        idx = np.arange(a, b, every)
        if every > 1 and (b - 1) not in idx:
            idx = np.append(idx, b - 1)
        blk = {"k": idx.tolist(), "t": [int(x) for x in self.t[idx]], "v": [r(x, 4) for x in self.v[idx]],
               "i": [r(x, 3) for x in self.i[idx]], "T": [r(x, 2) for x in self.T[idx]],
               "ref": [r(x, 5) for x in self.ref[idx]], "ekf_sd": [r(x, 5) for x in self.ekf_sd[idx]]}
        for key in self.keys:
            blk[key] = [r(x, 5) for x in self.out[key][idx]]
        return blk

    def stats(self):
        k = self.k
        n = getattr(self, "n_target", self.n)
        out = {"k": k, "n": n, "progress": r(k / n, 4), "cpu_c": cpu_temp_c(), "ms": {}, "rmse_pts": {},
               "max_pts": {}, "now_err_pts": {}}
        for key in self.keys:
            e = 100 * (self.out[key][:k] - self.ref[:k])
            if key in self.ms:
                ms = self.ms[key][:k]
                out["ms"][key] = {"median": r(np.median(ms), 4), "p99": r(np.percentile(ms, 99), 4)} if k else None
            else:
                out["ms"][key] = None
            out["rmse_pts"][key] = r(np.sqrt(np.mean(e**2)), 3) if k else None
            out["max_pts"][key] = r(np.max(np.abs(e)), 3) if k else None
            out["now_err_pts"][key] = r(e[-1], 3) if k else None
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycle", default="25degC_UDDS", choices=replay_names())
    ap.add_argument("--speed", type=float, default=10.0, help="samples per second (1 = real time; the data is 1 Hz)")
    ap.add_argument("--soc-error", type=float, default=20.0, help="initial SoC error of EKF and coulomb counting, points")
    ap.add_argument("--config", type=Path, default=ROOT / "demo" / "config.toml")
    ap.add_argument("--offline", action="store_true", help="do not connect to a broker")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many samples (0 = whole cycle)")
    ap.add_argument("--threads", type=int, default=1, help="ONNX Runtime threads per network")
    ap.add_argument("--history-every", type=float, default=HISTORY_EVERY_S, help="seconds between history messages")
    g = ap.add_argument_group("SMBus sources")
    g.add_argument("--source", choices=["replay", "smbus-emulated", "smbus", "bq27426-emulated", "bq27426"],
                   default="replay")
    g.add_argument("--bus", type=int, default=1, help="I2C bus of the real pack (Raspberry Pi: 1)")
    g.add_argument("--addr", type=lambda x: int(x, 0), default=sbs.SMART_BATTERY_ADDR)
    g.add_argument("--series", type=int, default=1, help="cells in series in the pack")
    g.add_argument("--parallel", type=int, default=1, help="cells in parallel in the pack")
    g.add_argument("--cell-voltage-cmd", type=lambda x: int(x, 0), default=None,
                   help="read the cell voltage from this code (mV word) instead of pack voltage / series")
    g.add_argument("--period", type=float, default=1.0, help="seconds between samples of a real pack")
    g.add_argument("--bus-error-rate", type=float, default=0.0, help="emulated bus error probability per transfer")
    args = ap.parse_args()

    cfg = {} if args.offline else load_config(args.config)
    topic = cfg.get("demo", {}).get("topic", "bms-soc-estimation/demo").rstrip("/")
    capacity, pack_info, reader = None, None, None
    if args.source == "replay":
        source = ReplaySource(args.cycle)
    elif args.source.startswith("bq27426"):
        from socest.bq27426 import Bq27426Reader, EmulatedBq27426

        em = EmulatedBq27426(args.cycle, error_rate=args.bus_error_rate) if args.source.endswith("emulated") else None
        if em is None:
            args.period = max(args.period, 1.0)       # one burst read per sample, at most 2 per second allowed
            args.speed = 1.0 / args.period
        reader = Bq27426Reader(em if em is not None else LinuxI2C(args.bus), min_interval_s=0.0 if em else 0.5)
        pack_info = reader.info()
        source = SMBusSource(reader, em, n=args.limit or 86400, period=args.period, family="bq27426")
        if em is None:
            print("BQ27426: the EKF and networks model an LG HG2 cell; the gauge must be configured for the cell "
                  "(design capacity, chemistry profile) with TI's tools", flush=True)
    else:
        em = None
        if args.source == "smbus-emulated":
            from socest.smart_battery_emulator import EmulatedSmartBattery

            em = EmulatedSmartBattery(args.cycle, error_rate=args.bus_error_rate, seed=0)
            transport = em
        else:
            transport = LinuxI2C(args.bus)
            args.speed = 1.0 / args.period
        reader = SmartBatteryReader(transport, addr=args.addr, retry_delay_s=0.0 if em is not None else 0.02)
        pack_info = reader.info()
        source = SMBusSource(reader, em, args.series, args.parallel, args.cell_voltage_cmd,
                             n=args.limit or 86400, period=args.period)
        if em is None:
            fcc = reader.read("FullChargeCapacity")[0]
            if not reader.scaling.capacity_mwh and fcc:
                capacity = fcc / args.parallel          # coulomb counting with the pack's own capacity
            print(f"pack: {pack_info.get('ManufacturerName')} {pack_info.get('DeviceName')} "
                  f"({pack_info.get('DeviceChemistry')}); the EKF and networks model an LG HG2 cell", flush=True)
    run = Run(source, args.soc_error, args.threads, n=args.limit or None, capacity_ah=capacity)
    if args.source == "bq27426" and source.last_sample and source.last_sample.get("FullChargeCapacity"):
        run.est["cc"] = StreamingCoulomb(source.last_sample["FullChargeCapacity"], soc0=run.est["cc"].soc)
    n = min(run.n, args.limit) if args.limit else run.n
    run.n_target = n
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + (
        args.cycle if args.source == "replay" else f"{args.source}-{source.name}".replace(" ", "-"))
    out_dir = ROOT / "demo" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    import onnxruntime

    meta = {"run_id": run_id, "device": cfg.get("demo", {}).get("device") or device_name(), "host": socket.gethostname(),
            "hardware": hardware_info(), "offline": args.offline,
            "broker": None if args.offline else cfg.get("broker", {}).get("host"),
            "cycle": source.name, "n_samples": n, "speed": args.speed, "soc_error_pts": args.soc_error,
            "source": source.kind, "live": getattr(source, "live", False), "keys": run.keys,
            "ref_label": source.ref_label, "pack": pack_info,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "labels": run.labels, "models": run.models, "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__, "publish_every_s": PUBLISH_EVERY_S}
    pub = Publisher(cfg, topic, args.offline)
    pub.send("summary", "", retain=True)          # clear the previous run's summary
    pub.send("history", "", retain=True)
    pub.send("meta", meta, retain=True, wait=True)
    pub.send("online", "1", retain=True)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *a: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *a: stop.update(flag=True))

    print(f"run {run_id}: {n} samples of {source.name} ({source.kind}) at {args.speed:g} samples/s "
          f"(about {n / args.speed / 60:.0f} min), EKF and coulomb counting start {args.soc_error:g} points low",
          flush=True)
    t_start = time.time()
    last_pub, last_hist, last_print, sent = t_start, t_start, t_start, 0
    while run.k < n and not stop["flag"]:
        target = t_start + run.k / args.speed
        delay = target - time.time()
        if delay > 0:
            time.sleep(delay)
        run.step()
        now = time.time()
        if now - last_pub >= PUBLISH_EVERY_S or run.k == n:
            msg = run.block(sent, run.k)
            msg["run_id"], msg["stats"] = run_id, run.stats()
            pub.send("telemetry", msg)
            sent, last_pub = run.k, now
        if now - last_hist >= args.history_every:
            pub.send("history", {"run_id": run_id, **run.block(0, run.k, max(1, math.ceil(run.k / HISTORY_POINTS))),
                                 "stats": run.stats()}, retain=True)
            last_hist = now
        if now - last_print >= 10 or run.k == n:
            s = run.stats()
            k = run.k - 1
            est = " | ".join(f"{key} {100 * run.out[key][k]:5.1f}" for key in run.keys)
            ms = " ".join(f"{key} {s['ms'][key]['median']:.2f}" for key in KEYS)
            print(f"{run.k:6d}/{n}  SoC % ref {100 * run.ref[k]:5.1f} | {est}  ms/step {ms}  CPU {s['cpu_c']} C",
                  flush=True)
            last_print = now

    k = run.k
    stats = run.stats()
    summary = {"run_id": run_id, "finished": not stop["flag"], "samples": k, "wall_s": round(time.time() - t_start, 1),
               "rmse_pts": stats["rmse_pts"], "max_pts": stats["max_pts"], "ms": stats["ms"], "cpu_c_end": cpu_temp_c(),
               "labels": run.labels, "device": meta["device"], "hardware_model": meta["hardware"]["model"],
               "code_commit": meta["hardware"]["code_commit"], "power_end": power_status()}
    if reader is not None:
        b = reader.stats
        summary["smbus"] = {"reads_ok": b.ok, "retried": b.retried, "failed": b.failed, "pec_errors": b.pec_errors,
                            "stale_samples": source.stale,
                            "read_ms_median": r(np.median(source.read_ms), 3) if source.read_ms else None}
    pub.send("history", {"run_id": run_id, **run.block(0, k, max(1, math.ceil(k / HISTORY_POINTS))), "stats": stats},
             retain=True)
    pub.send("summary", summary, retain=True, wait=True)
    pub.close()

    import pandas as pd

    log = pd.DataFrame({"time_s": run.t[:k], "voltage_v": run.v[:k], "current_a": run.i[:k], "temp_degc": run.T[:k],
                        "soc_ref": run.ref[:k], **{f"soc_{key}": run.out[key][:k] for key in run.keys},
                        "ekf_soc_std": run.ekf_sd[:k], **{f"ms_{key}": run.ms[key][:k] for key in KEYS}})
    log.to_csv(out_dir / "samples.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{'SoC error over the run (points)':58s}RMSE    max    ms/step median   p99")
    for key in run.keys:
        m = stats["ms"][key]
        ms = f"{m['median']:7.3f}  {m['p99']:7.3f}" if m else "      -        -"
        print(f"  {run.labels[key]:56s} {stats['rmse_pts'][key]:6.2f} {stats['max_pts'][key]:6.2f}        {ms}")
    if "smbus" in summary:
        print(f"  Bus: {summary['smbus']}")
    if getattr(source, "live", False):
        print("  reference = the gauge's own SoC; EKF and networks are LG HG2 models")
    if hasattr(reader, "sign_mismatch") and reader.sign_mismatch:
        print(f"  WARNING: {reader.sign_mismatch} samples where the DSG flag contradicts the current sign")
    print(f"\nlog: {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
