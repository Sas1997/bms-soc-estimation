"""TI BQ27426 system-side gauge: decoding, one rate-limited burst per sample, emulator, demo integration."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from socest.bq27426 import (ADDR, BURST_END, BURST_START, Bq27426Reader, EmulatedBq27426, consistent,
                            decode_burst)

ROOT = Path(__file__).resolve().parents[1]


def burst_bytes(words: dict) -> bytes:
    mem = bytearray(BURST_END - BURST_START + 1)
    for code, w in words.items():
        w &= 0xFFFF
        mem[code - BURST_START], mem[code - BURST_START + 1] = w & 0xFF, w >> 8      # little-endian pairs
    return bytes(mem)


def test_decode_standard_commands():
    d = decode_burst(burst_bytes({0x02: 2982, 0x04: 3712, 0x06: (1 << 9) | (1 << 4) | (1 << 1), 0x0C: 1830,
                                  0x0E: 2950, 0x10: -1500, 0x18: -5568, 0x1C: 62, 0x20: 97 | 3 << 8, 0x30: 61}))
    assert d["Temperature"] == pytest.approx(25.05)
    assert d["Voltage"] == pytest.approx(3.712)
    assert d["Flags"] == ["FC", "BAT_DET", "DSG"]
    assert d["RemainingCapacity"] == pytest.approx(1.83) and d["FullChargeCapacity"] == pytest.approx(2.95)
    assert d["AverageCurrent"] == pytest.approx(-1.5) and d["AveragePower"] == pytest.approx(-5.568)
    assert d["StateOfCharge"] == 62 and d["StateOfChargeUnfiltered"] == 61
    assert d["StateOfHealth"] == 97 and d["StateOfHealthStatus"] == "learned Qmax"


class FakeClock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def test_one_transfer_per_sample_and_at_most_two_per_second():
    em = EmulatedBq27426("10degC_US06")
    clock = FakeClock()
    r = Bq27426Reader(em, clock=clock)
    times = []
    for _ in range(20):
        r.sample()
        times.append(r._last)
        em.advance()
        clock.t += 0.1                       # caller polls far too fast
    assert em.max_transfers_per_step == 1
    assert np.min(np.diff(times)) >= 0.5 - 1e-12


def test_reader_matches_recording_and_signs():
    em = EmulatedBq27426("10degC_US06")
    r = Bq27426Reader(em, min_interval_s=0)
    for _ in range(300):
        k = em.k
        s = r.sample()
        assert s["Voltage"] == pytest.approx(em.v[k], abs=5e-4 + 1e-9)
        assert s["Current"] == pytest.approx(em.i[k], abs=5e-4 + 1e-9)      # positive = discharge
        assert s["Temperature"] == pytest.approx(em.T[k], abs=0.05 + 1e-9)
        em.advance()
    assert r.sign_mismatch == 0 and r.stats.failed == 0


def test_nack_above_0x6b_and_wrong_address():
    em = EmulatedBq27426("10degC_US06")
    with pytest.raises(OSError):
        em.read(ADDR, 0x6A, 4)
    with pytest.raises(OSError):
        em.read(0x0B, 0x02, 2)
    r = Bq27426Reader(em, addr=0x0B, min_interval_s=0)
    s = r.sample()
    assert s["Voltage"] is None and r.stats.failed == 1


def test_sign_check_flags_a_flipped_current():
    log = pd.DataFrame({"Current": [1.0, 1.2, -0.8, 0.0], "Flags": ["DSG", "DSG", "", ""]})
    assert consistent(log)["current_sign_agrees_with_DSG"] == 1.0
    log["Current"] *= -1
    assert consistent(log)["current_sign_agrees_with_DSG"] == 0.0


def test_edge_bq27426_sources():
    pytest.importorskip("onnxruntime")
    sys.path.insert(0, str(ROOT / "demo"))
    import edge

    em = EmulatedBq27426("10degC_US06")
    run = edge.Run(edge.SMBusSource(Bq27426Reader(em, min_interval_s=0), em, family="bq27426"), 20.0, 1)
    for _ in range(100):
        run.step()
    assert run.src.kind == "bq27426-emulated" and "gauge" in run.keys
    assert abs(run.out["gauge"][99] - run.ref[99]) < 0.02

    em2 = EmulatedBq27426("10degC_US06")       # as a live gauge: no emulator handle
    live = edge.SMBusSource(Bq27426Reader(em2, min_interval_s=0), None, n=20, family="bq27426")
    run2 = edge.Run(live, 0.0, 1)
    for _ in range(5):
        run2.step()
        em2.advance()
    assert live.kind == "bq27426" and "gauge" not in run2.keys
    np.testing.assert_allclose(run2.ref[:5], run2.gauge[:5])


def test_logger_cli_bq27426(tmp_path):
    out = subprocess.run([sys.executable, str(ROOT / "demo" / "smbus_log.py"), "--gauge", "bq27426", "--emulate",
                          "10degC_US06", "--duration", "300", "--out", str(tmp_path)],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    log = pd.read_csv(tmp_path / "log.csv")
    assert len(log) == 300 and {"StateOfCharge", "StateOfHealth", "FullChargeCapacity", "Flags"} <= set(log.columns)
    assert "'current_sign_agrees_with_DSG': 1.0" in out.stdout
