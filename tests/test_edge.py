"""Edge demo: ONNX networks, the edge run loop and its MQTT messages (no broker needed)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("onnxruntime")
sys.path.insert(0, str(ROOT / "demo"))

from socest.estimators import selected_networks  # noqa: E402
from socest.onnx_estimator import OnnxNetwork  # noqa: E402
from socest.replay import load_replay, replay_names  # noqa: E402


def test_replay_files_are_test_cycles_with_valid_columns():
    assert set(replay_names()) == {"25degC_UDDS", "n20degC_UDDS", "10degC_US06"}
    for name in replay_names():
        u = load_replay(name)
        assert list(u.columns) == ["time_s", "voltage_v", "current_a", "temp_degc", "soc_ref"]
        assert (np.diff(u.time_s) == 1).all()                      # 1 Hz
        assert u.soc_ref.iloc[0] == pytest.approx(1.0, abs=1e-6)  # full-charge start
        assert 2.5 < u.voltage_v.min() and u.voltage_v.max() < 4.25
        assert u.current_a.mean() > 0                              # net discharge, positive = discharge


@pytest.mark.parametrize("kind", ["lstm", "transformer"])
def test_onnx_step_by_step_matches_batch(kind):
    u = load_replay("10degC_US06").iloc[:400]
    net = OnnxNetwork(selected_networks()[kind])
    batch = net(u.voltage_v, u.current_a, u.temp_degc)
    steps = [net.step(a, b, c) for a, b, c in zip(u.voltage_v, u.current_a, u.temp_degc)]
    np.testing.assert_allclose(steps, batch, atol=1e-5)
    assert 0.0 < batch.min() and batch.max() < 1.1


def test_edge_run_and_messages():
    import edge

    run = edge.Run(edge.ReplaySource("10degC_US06"), soc_error_pts=20.0, threads=1)
    run.n_target = 300
    for _ in range(300):
        run.step()
    assert run.out["cc"][0] == pytest.approx(run.ref[0] - 0.20)    # coulomb counting keeps the initial error
    assert abs(run.out["ekf"][299] - run.ref[299]) < 0.05           # the EKF has corrected most of it
    blk = run.block(100, 300)
    assert blk["k"][0] == 100 and len(blk["t"]) == 200 and set(edge.KEYS) <= set(blk)
    hist = run.block(0, 300, every=7)
    assert hist["k"][-1] == 299                                     # downsampled history keeps the latest sample
    s = run.stats()
    assert s["progress"] == 1.0 and set(s["rmse_pts"]) == set(edge.KEYS)
    json.dumps(blk), json.dumps(s)                                  # serialisable, no NaN


def test_edge_offline_cli(tmp_path):
    out = subprocess.run([sys.executable, str(ROOT / "demo" / "edge.py"), "--offline", "--cycle", "10degC_US06",
                          "--limit", "200", "--speed", "100000"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    run_dir = ROOT / out.stdout.strip().splitlines()[-1].split("log: ")[1].strip()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["finished"] and summary["samples"] == 200
    for f in ("samples.csv", "meta.json", "summary.json"):
        (run_dir / f).unlink()
    run_dir.rmdir()
