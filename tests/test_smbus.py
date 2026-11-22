"""SMBus smart-battery path: SBS decoding, PEC, the read-only reader, the emulator and register identification."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from socest import register_id, sbs
from socest.smart_battery_emulator import VENDOR_TRUTH, EmulatedSmartBattery
from socest.smbus_link import PECError, SmartBatteryReader, cell_inputs

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- SBS
def test_pec_is_crc8_smbus():
    assert sbs.crc8(b"123456789") == 0xF4                  # CRC-8/SMBUS check value
    assert sbs.crc8(b"") == 0


def test_decode_units_and_sign_convention():
    assert sbs.decode("Voltage", 15741) == pytest.approx(15.741)
    assert sbs.decode("Current", 0xFF9C) == pytest.approx(0.1)          # -100 mA in SBS = 0.1 A discharge here
    assert sbs.decode("Current", 250) == pytest.approx(-0.25)           # charging is negative here
    assert sbs.decode("Temperature", 2982) == pytest.approx(25.05)
    assert sbs.decode("RunTimeToEmpty", 0xFFFF) is None
    assert sbs.decode("RemainingCapacity", 2500) == pytest.approx(2.5)
    assert sbs.decode("ManufactureDate", (2026 - 1980) * 512 + 11 * 32 + 21) == "2026-11-21"
    assert sbs.decode("DeviceChemistry", b"LION\x00") == "LION"


def test_specification_info_scaling_and_capacity_mode():
    info = sbs.specification_info(0x1031)                 # IPScale 1, VScale 0, SBS 1.1 with PEC
    assert info == {"revision": 1, "version": "1.1 with PEC", "pec": True, "vscale": 0, "ipscale": 1}
    sc = sbs.Scaling.from_words(0x0131, 0x8000)           # VScale 1, capacities in 10 mWh
    assert sbs.decode("Voltage", 1574, sc) == pytest.approx(15.74)
    assert sbs.decode("RemainingCapacity", 4000, sc) == pytest.approx(40.0)   # Wh
    assert sbs.decode("BatteryStatus", 0x00C0)["flags"] == ["INITIALIZED", "DISCHARGING"]
    assert sbs.decode("BatteryStatus", 0x0003)["error"] == "UnsupportedCommand"


# ---------------------------------------------------------------- reader + emulator
class Recorder:
    """Transport wrapper that records every transfer."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def read(self, addr, cmd, n):
        self.calls.append((addr, cmd, n))
        return self.inner.read(addr, cmd, n)


def test_reader_only_issues_read_transfers():
    assert not [m for m in dir(SmartBatteryReader) if "write" in m.lower()]
    rec = Recorder(EmulatedSmartBattery("10degC_US06"))
    r = SmartBatteryReader(rec)
    r.info(), r.sample(scan=range(0x30, 0x40))
    assert rec.calls and all(a == sbs.SMART_BATTERY_ADDR and 0 <= c <= 0xFF for a, c, _ in rec.calls)


def test_emulated_pack_reads_back_the_recording_at_sbs_resolution():
    em = EmulatedSmartBattery("10degC_US06")
    r = SmartBatteryReader(em)
    assert r.pec and r.info()["DeviceChemistry"] == "LION"
    for _ in range(50):
        k = em.k
        s = r.sample()
        assert s["Voltage"] == pytest.approx(em.v[k], abs=5e-4 + 1e-9)
        assert s["Current"] == pytest.approx(em.i[k], abs=5e-4 + 1e-9)
        assert s["Temperature"] == pytest.approx(em.T[k], abs=0.05 + 1e-9)
        assert 0 <= s["RelativeStateOfCharge"] <= 100
        em.advance()


def test_pec_catches_corruption_and_retries_recover():
    em = EmulatedSmartBattery("10degC_US06", corrupt_rate=0.1, error_rate=0.1, seed=3)
    r = SmartBatteryReader(em, pec=True, retries=8, retry_delay_s=0.0)
    for _ in range(100):
        s = r.sample()
        assert s["Voltage"] == pytest.approx(em.v[em.k], abs=5e-4 + 1e-9)    # never a corrupted value
        em.advance()
    assert r.stats.pec_errors > 0 and r.stats.failed == 0
    bad = EmulatedSmartBattery("10degC_US06", corrupt_rate=1.0)
    with pytest.raises(PECError):
        SmartBatteryReader(bad, pec=True, retries=1, retry_delay_s=0.0)


def test_wrong_address_and_unsupported_command_fail_like_a_nack():
    em = EmulatedSmartBattery("10degC_US06")
    with pytest.raises(OSError):
        SmartBatteryReader(em, addr=0x0C, retries=0)
    r = SmartBatteryReader(em)
    assert r.try_word(0x55, retries=0) is None


def test_cell_inputs_for_a_pack():
    s = {"Voltage": 15.6, "Current": 3.0, "Temperature": 30.0}
    assert cell_inputs(s, n_series=4, n_parallel=2) == (pytest.approx(3.9), pytest.approx(1.5), 30.0)
    assert cell_inputs(s, n_series=4, cell_voltage=3.95)[0] == 3.95


# ---------------------------------------------------------------- register identification
def test_register_identification_on_synthetic_log():
    rng = np.random.default_rng(0)
    n = 2000
    v = 3.6 + 0.4 * np.sin(np.arange(n) / 150) + rng.normal(0, 0.01, n)
    i = 2 * np.sin(np.arange(n) / 37) + 0.5
    T = -5 + 3 * np.sin(np.arange(n) / 400)
    log = pd.DataFrame({"t_s": np.arange(n, dtype=float), "Voltage": np.round(v, 3), "Current": np.round(i, 3),
                        "Temperature": np.round(T, 1)})
    log["raw_0x40"] = np.round(v * 500).astype(int)                         # 2 mV per count
    log["raw_0x41"] = np.round(-i * 100).astype(int) & 0xFFFF                # -10 mA, signed
    log["raw_0x42"] = np.round(T * 10).astype(int) & 0xFFFF                  # 0.1 degC, signed, below zero
    log["raw_0x43"] = 7
    log["raw_0x44"] = rng.integers(0, 65535, n)                             # noise
    log["raw_0x45"] = None
    t = register_id.identify(log).set_index("code")
    assert t.loc["0x40", "tracks"] == "Voltage" and t.loc["0x40", "scale"] == pytest.approx(0.002)
    assert t.loc["0x41", "tracks"] == "Current" and t.loc["0x41", "scale"] == pytest.approx(-0.01)
    assert t.loc["0x41", "type"] == "i16"
    assert t.loc["0x42", "tracks"] == "Temperature" and t.loc["0x42", "scale"] == pytest.approx(0.1)
    assert t.loc["0x42", "type"] == "i16" and abs(t.loc["0x42", "offset"]) < 0.1
    assert t.loc["0x40", "verdict"] == t.loc["0x41", "verdict"] == t.loc["0x42", "verdict"] == "match"
    assert t.loc["0x43", "verdict"] == "constant"
    assert t.loc["0x44", "verdict"] == "unexplained"
    assert t.loc["0x45", "verdict"] == "no answer"


def test_identifies_the_emulator_vendor_registers():
    em = EmulatedSmartBattery("10degC_US06")
    r = SmartBatteryReader(em)
    rows = []
    while not em.done:
        s = r.sample(scan=sorted(VENDOR_TRUTH))
        s["t_s"] = float(em.t[em.k])
        rows.append(s)
        em.advance()
    t = register_id.identify(pd.DataFrame(rows)).set_index("code")
    assert t.loc["0x3f", "tracks"] == "Voltage" and t.loc["0x3f", "scale"] == pytest.approx(0.001)
    assert t.loc["0x31", "tracks"] == "Current" and t.loc["0x31", "scale"] == pytest.approx(0.01)
    assert t.loc["0x32", "tracks"] == "discharged_mah"
    assert t.loc["0x30", "tracks"] == "Temperature"
    assert set(t.loc[["0x33", "0x3c", "0x3d", "0x3e"], "verdict"]) == {"constant"}


# ---------------------------------------------------------------- demo integration
def test_edge_smbus_sources():
    pytest.importorskip("onnxruntime")
    sys.path.insert(0, str(ROOT / "demo"))
    import edge

    em = EmulatedSmartBattery("10degC_US06")
    run = edge.Run(edge.SMBusSource(SmartBatteryReader(em), em), soc_error_pts=20.0, threads=1)
    for _ in range(200):
        run.step()
    assert "gauge" in run.keys and run.ref[0] == pytest.approx(1.0)
    assert abs(run.out["gauge"][199] - run.ref[199]) < 0.02

    # live path (no emulator handle): the pack's gauge becomes the reference
    em2 = EmulatedSmartBattery("10degC_US06")
    live = edge.SMBusSource(SmartBatteryReader(em2), None, n=50)
    run2 = edge.Run(live, soc_error_pts=0.0, threads=1)
    for _ in range(10):
        run2.step()
        em2.advance()
    assert "gauge" not in run2.keys
    np.testing.assert_allclose(run2.ref[:10], run2.gauge[:10])


def test_smbus_logger_cli(tmp_path):
    out = subprocess.run([sys.executable, str(ROOT / "demo" / "smbus_log.py"), "--emulate", "10degC_US06",
                          "--duration", "600", "--identify", "--out", str(tmp_path)],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    log = pd.read_csv(tmp_path / "log.csv")
    assert len(log) == 600 and {"Voltage", "Current", "raw_0x3f"} <= set(log.columns)
    assert "0x3f: match Voltage" in out.stdout
    assert (tmp_path / "registers.md").exists() and (tmp_path / "info.json").exists()
