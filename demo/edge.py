"""Edge node of the cloud-connected BMS demo: estimate SoC on the device, stream to the cloud.

Replays a test cycle from demo/replay/ one sample at a time, as if the
measurements came from a live cell, and runs four estimators on every sample:

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
import platform
import signal
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from socest.estimators import EKFEstimator, selected_networks  # noqa: E402
from socest.onnx_estimator import OnnxNetwork  # noqa: E402
from socest.replay import load_replay, replay_names  # noqa: E402
from socest.streaming import StreamingCoulomb, StreamingEKF  # noqa: E402

KEYS = ["ekf", "lstm", "transformer", "cc"]
LABELS = {"ekf": "EKF, 3RC (calibrated)", "lstm": "LSTM", "transformer": "Transformer", "cc": "Coulomb counting"}
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


# ---------------------------------------------------------------- run
class Run:
    def __init__(self, cycle: str, soc_error_pts: float, threads: int):
        u = load_replay(cycle)
        self.cycle = cycle
        self.t = u.time_s.to_numpy(float)
        self.v = u.voltage_v.to_numpy(float)
        self.i = u.current_a.to_numpy(float)
        self.T = u.temp_degc.to_numpy(float)
        self.ref = u.soc_ref.to_numpy(float)
        self.n = len(self.t)
        soc0 = float(self.ref[0]) - soc_error_pts / 100.0
        nets = selected_networks()
        ekf = EKFEstimator.tuned(None, label=LABELS["ekf"])
        self.models = {"ekf": ekf.variant, "lstm": nets["lstm"], "transformer": nets["transformer"]}
        self.est = {
            "ekf": StreamingEKF(ekf, temp0=float(self.T[0]), soc0=soc0),
            "lstm": OnnxNetwork(nets["lstm"], threads=threads),
            "transformer": OnnxNetwork(nets["transformer"], threads=threads),
            "cc": StreamingCoulomb(ekf.cap_of(float(self.T[0])), soc0=soc0),
        }
        self.out = {k: np.full(self.n, np.nan) for k in KEYS}
        self.ekf_sd = np.full(self.n, np.nan)
        self.ms = {k: np.full(self.n, np.nan) for k in KEYS}
        self.k = 0

    def step(self):
        k = self.k
        v, i, T = self.v[k], self.i[k], self.T[k]
        for key, est in self.est.items():
            t0 = time.perf_counter_ns()
            self.out[key][k] = est.step(v, i, T)
            self.ms[key][k] = (time.perf_counter_ns() - t0) / 1e6
        self.ekf_sd[k] = self.est["ekf"].soc_std
        self.k += 1

    def block(self, a, b, every=1):
        idx = np.arange(a, b, every)
        if every > 1 and (b - 1) not in idx:
            idx = np.append(idx, b - 1)
        blk = {"k": idx.tolist(), "t": [int(x) for x in self.t[idx]], "v": [r(x, 4) for x in self.v[idx]],
               "i": [r(x, 3) for x in self.i[idx]], "T": [r(x, 2) for x in self.T[idx]],
               "ref": [r(x, 5) for x in self.ref[idx]], "ekf_sd": [r(x, 5) for x in self.ekf_sd[idx]]}
        for key in KEYS:
            blk[key] = [r(x, 5) for x in self.out[key][idx]]
        return blk

    def stats(self):
        k = self.k
        n = getattr(self, "n_target", self.n)
        out = {"k": k, "n": n, "progress": r(k / n, 4), "cpu_c": cpu_temp_c(), "ms": {}, "rmse_pts": {},
               "max_pts": {}, "now_err_pts": {}}
        for key in KEYS:
            ms = self.ms[key][:k]
            e = 100 * (self.out[key][:k] - self.ref[:k])
            out["ms"][key] = {"median": r(np.median(ms), 4), "p99": r(np.percentile(ms, 99), 4)} if k else None
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
    args = ap.parse_args()

    cfg = {} if args.offline else load_config(args.config)
    topic = cfg.get("demo", {}).get("topic", "bms-soc-estimation/demo").rstrip("/")
    run = Run(args.cycle, args.soc_error, args.threads)
    n = min(run.n, args.limit) if args.limit else run.n
    run.n_target = n
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.cycle
    out_dir = ROOT / "demo" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    import onnxruntime

    meta = {"run_id": run_id, "device": cfg.get("demo", {}).get("device") or device_name(), "host": socket.gethostname(),
            "cycle": args.cycle, "n_samples": n, "speed": args.speed, "soc_error_pts": args.soc_error,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "labels": LABELS, "models": run.models, "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__, "publish_every_s": PUBLISH_EVERY_S}
    pub = Publisher(cfg, topic, args.offline)
    pub.send("summary", "", retain=True)          # clear the previous run's summary
    pub.send("history", "", retain=True)
    pub.send("meta", meta, retain=True, wait=True)
    pub.send("online", "1", retain=True)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *a: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *a: stop.update(flag=True))

    print(f"run {run_id}: {n} samples of {args.cycle} at {args.speed:g} samples/s "
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
            est = " | ".join(f"{key} {100 * run.out[key][k]:5.1f}" for key in KEYS)
            ms = " ".join(f"{key} {s['ms'][key]['median']:.2f}" for key in KEYS)
            print(f"{run.k:6d}/{n}  SoC % ref {100 * run.ref[k]:5.1f} | {est}  ms/step {ms}  CPU {s['cpu_c']} C",
                  flush=True)
            last_print = now

    k = run.k
    stats = run.stats()
    summary = {"run_id": run_id, "finished": not stop["flag"], "samples": k, "wall_s": round(time.time() - t_start, 1),
               "rmse_pts": stats["rmse_pts"], "max_pts": stats["max_pts"], "ms": stats["ms"], "cpu_c_end": cpu_temp_c(),
               "labels": LABELS, "device": meta["device"]}
    pub.send("history", {"run_id": run_id, **run.block(0, k, max(1, math.ceil(k / HISTORY_POINTS))), "stats": stats},
             retain=True)
    pub.send("summary", summary, retain=True, wait=True)
    pub.close()

    import pandas as pd

    log = pd.DataFrame({"time_s": run.t[:k], "voltage_v": run.v[:k], "current_a": run.i[:k], "temp_degc": run.T[:k],
                        "soc_ref": run.ref[:k], **{f"soc_{key}": run.out[key][:k] for key in KEYS},
                        "ekf_soc_std": run.ekf_sd[:k], **{f"ms_{key}": run.ms[key][:k] for key in KEYS}})
    log.to_csv(out_dir / "samples.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\nSoC error over the run (points)        RMSE    max    ms/step median   p99")
    for key in KEYS:
        m = stats["ms"][key]
        print(f"  {LABELS[key]:34s} {stats['rmse_pts'][key]:6.2f} {stats['max_pts'][key]:6.2f}"
              f"        {m['median']:7.3f}  {m['p99']:7.3f}")
    print(f"\nlog: {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
