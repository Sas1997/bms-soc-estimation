"""Collect a verifiable record of the edge runs on this device, ready to commit to the repository.

Run on the Raspberry Pi (or any edge device) after the demo runs you want to keep:

    python demo/collect_evidence.py

It writes results/edge/<device>/<UTC date>/ with:

    README.md        what ran, where, with which code; generated from the files below
    system.json      hardware read from the system (/proc/device-tree/model, /etc/os-release,
                     kernel, memory, CPU count), power report (vcgencmd), code commit, installed packages
    tests.txt        output of the test suite run on the device
    cost.csv         per-step compute time and memory of every estimator, measured on the device
    runs/<run_id>/   meta.json, summary.json and samples.csv of every edge run in demo/runs/ from the
                     last --days days (the offline check run it makes itself included), the same
                     run IDs the live dashboard shows
    sha256sums.txt   checksums of every file above (verify with: sha256sum -c sha256sums.txt)

Nothing secret is copied: demo/config.toml (broker login) is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT / "src"))

import edge  # noqa: E402  (hardware_info, power_status)


def run(cmd, timeout=900):
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return out.returncode, out.stdout + out.stderr


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=7, help="include edge runs from the last N days (default 7)")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-cost", action="store_true")
    ap.add_argument("--skip-check-run", action="store_true", help="do not make the offline check run")
    args = ap.parse_args()

    hw = edge.hardware_info()
    now = datetime.now(timezone.utc)
    out = ROOT / "results" / "edge" / slug(hw["model"]) / now.strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {hw['model']}  |  {hw.get('os')}  |  kernel {hw['kernel']}  |  code {hw['code_commit']}")
    print(f"writing {out.relative_to(ROOT)}/", flush=True)

    _, freeze = run([sys.executable, "-m", "pip", "freeze"])
    system = {"collected_utc": now.isoformat(timespec="seconds"), "hostname": edge.socket.gethostname(),
              **hw, "packages": sorted(l for l in freeze.splitlines() if "==" in l)}
    (out / "system.json").write_text(json.dumps(system, indent=1))

    tests_line = "not run"
    if not args.skip_tests:
        print("running the test suite ...", flush=True)
        code, text = run([sys.executable, "-m", "pytest", "-q"])
        (out / "tests.txt").write_text(f"$ python -m pytest -q   (exit code {code})\n\n{text}")
        tests_line = next((l.strip() for l in reversed(text.splitlines()) if "passed" in l or "failed" in l), "?")
        print(f"  {tests_line}", flush=True)

    if not args.skip_cost:
        print("measuring compute cost per estimator ...", flush=True)
        code, text = run([sys.executable, "experiments/m4_cost.py", "--label", slug(hw["model"]),
                          "--out", str(out / "cost.csv")])
        if code != 0:
            print(text)

    if not args.skip_check_run:
        print("offline check run (10 degC US06, held-out temperature) ...", flush=True)
        run([sys.executable, "demo/edge.py", "--offline", "--cycle", "10degC_US06", "--speed", "1000"])

    runs_dir = ROOT / "demo" / "runs"
    since = now - timedelta(days=args.days)
    kept = []
    for d in sorted(runs_dir.glob("*")) if runs_dir.exists() else []:
        meta_f, summ_f = d / "meta.json", d / "summary.json"
        if not (meta_f.exists() and summ_f.exists()):
            continue
        meta, summ = json.loads(meta_f.read_text()), json.loads(summ_f.read_text())
        started = datetime.fromisoformat(meta["started_utc"])
        if started < since:
            continue
        dst = out / "runs" / d.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("meta.json", "summary.json", "samples.csv"):
            if (d / f).exists():
                shutil.copy2(d / f, dst / f)
        kept.append((meta, summ))
    print(f"copied {len(kept)} edge runs", flush=True)

    # ---------------------------------------------------------------- README
    md = [f"# Edge runs on {hw['model']}", "",
          f"Collected {now.strftime('%Y-%m-%d %H:%M UTC')} by `demo/collect_evidence.py` on the device itself.", "",
          "| | |", "|:--|:--|",
          f"| Board | {hw['model']} |", f"| Operating system | {hw.get('os')} |", f"| Kernel | {hw['kernel']} |",
          f"| CPU cores / memory | {hw['cpus']} / {hw.get('memory_gb')} GB |", f"| Python | {hw['python']} |",
          f"| Code commit | `{hw['code_commit']}` |",
          (f"| Power | input {hw['ext5v_v']} V; throttled flags {hw['throttled']} (0x0 = no under-voltage or "
           "throttling since boot) |" if hw.get("ext5v_v") is not None else "| Power | not reported on this device |"),
          f"| Test suite on the device | {tests_line} |", ""]
    cost_f = out / "cost.csv"
    if cost_f.exists():
        import pandas as pd

        c = pd.read_csv(cost_f)
        md += ["## Compute cost on the device (one thread, 1,000 steps)", "",
               "| Estimator | ms per step, median | 99th percentile | Peak memory (MB) |", "|:--|--:|--:|--:|"]
        md += [f"| {r.estimator} | {r.ms_per_step_median:.3f} | {r.ms_per_step_p99:.3f} | {r.peak_rss_mb:.0f} |"
               for r in c.itertuples()]
        md += [""]
    if kept:
        keys = ["ekf", "lstm", "transformer", "cc"]
        md += ["## Edge runs", "",
               "Run IDs are the ones the live dashboard shows. RMSE in SoC percentage points against the reference; "
               "ms is the median compute time per 1 Hz step on the device.", "",
               "| Run ID | Cycle | Mode | Samples | RMSE EKF / LSTM / Transformer / CC | ms EKF / LSTM / Transformer | CPU at end |",
               "|:--|:--|:--|--:|:--|:--|--:|"]
        for meta, summ in kept:
            if "offline" not in meta:
                mode = "not recorded (run made before this field existed)"
            elif meta["offline"]:
                mode = "offline"
            else:
                mode = f"streamed to {meta.get('broker') or 'the cloud broker'}"
            if meta.get("source", "replay") != "replay":
                mode += f", source {meta['source']}"
            rm = " / ".join(f"{summ['rmse_pts'].get(k, float('nan')):.2f}" for k in keys if k in summ["rmse_pts"])
            ms = " / ".join(f"{summ['ms'][k]['median']:.2f}" for k in keys[:3] if summ["ms"].get(k))
            md += [f"| `{meta['run_id']}` | {meta['cycle']} | {mode} | {summ['samples']} | {rm} | {ms} | "
                   f"{'-' if summ.get('cpu_c_end') is None else str(summ['cpu_c_end']) + ' °C'} |"]
        md += [""]
    md += ["## Verify", "", "Every file here is listed with its SHA-256 checksum in `sha256sums.txt`:", "",
           "```", "sha256sum -c sha256sums.txt", "```", ""]
    (out / "README.md").write_text("\n".join(md))

    sums = []
    for f in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "sha256sums.txt"):
        sums.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out).as_posix()}")
    (out / "sha256sums.txt").write_text("\n".join(sums) + "\n")

    rel = out.relative_to(ROOT).as_posix()
    addr = (edge._cmd("hostname", "-I") or "").split()
    user = edge._cmd("whoami") or "ali"
    print(f"\ndone: {rel}/  ({len(sums)} files)")
    print("To copy it into the repository on the laptop, run these in PowerShell inside the repository folder:")
    print(f"  mkdir -Force {Path(rel).parent.as_posix()}")
    print(f"  scp -r {user}@{addr[0] if addr else edge.socket.gethostname()}:~/bms-soc-estimation/{rel} "
          f"{Path(rel).parent.as_posix()}/")


if __name__ == "__main__":
    main()
