"""TI BQ27426: system-side Impedance Track fuel gauge, read-only over I2C.

The BQ27426 sits on the system board next to a single Li-ion cell and measures
the cell's voltage, current (sense resistor) and temperature. It runs TI's
Impedance Track algorithm: an OCV table and a resistance table from one of
three pre-programmed chemistry profiles, coulomb counting, and learning of the
cell's chemical capacity (Qmax) and resistance during use.

It is not an SBS smart battery: it answers at 7-bit I2C address 0x55 with TI's
own "standard commands", 2-byte little-endian values at fixed command codes.

Source: TI, bq27426 Technical Reference, SLUUBB0 (December 2015):
    Table 5-1 standard commands, Table 5-4 Flags(), section 3.3 I2C command waiting time.

Read-only and rate-limited:
    * Only the standard read path is used (address, command code, repeated start,
      read). Control() subcommands are never used because they need a data write.
      Configuration (design capacity, chemistry profile) lives in data memory and
      is changed with TI's tools, not here.
    * SLUUBB0 section 3.3: "the host must not issue any standard command more than
      two times per second", or the gauge's watchdog can reset it. Every sample is
      therefore ONE incremental read of 0x02-0x31 (the address pointer
      auto-increments), and the reader enforces at least 0.5 s between transfers.
    * There is no PEC on this gauge; each sample is checked for plausibility instead.

Sign: AverageCurrent() is signed mA. SLUUBB0 does not state its sign in words;
AveragePower() "is negative during discharge and positive during charge", and the
reader assumes the same for current. It counts samples where the DSG flag
contradicts that (sign_mismatch), so a wrong assumption shows up in the log.
This package uses positive = discharge, so the decoded current is negated.
"""

from __future__ import annotations

import time

import numpy as np

from .smbus_link import ReadStats

ADDR = 0x55
BURST_START, BURST_END = 0x02, 0x31           # inclusive; one transfer per sample
MAX_CODE = 0x6B                               # reads above this are NACKed (datasheet)
MIN_INTERVAL_S = 0.5

# code: (name, kind, unit). kind: u16 / i16 / flags / soh
STANDARD = {
    0x02: ("Temperature", "u16", "0.1K"),
    0x04: ("Voltage", "u16", "mV"),
    0x06: ("Flags", "flags", ""),
    0x08: ("NominalAvailableCapacity", "u16", "mAh"),
    0x0A: ("FullAvailableCapacity", "u16", "mAh"),
    0x0C: ("RemainingCapacity", "u16", "mAh"),
    0x0E: ("FullChargeCapacity", "u16", "mAh"),
    0x10: ("AverageCurrent", "i16", "mA"),
    0x18: ("AveragePower", "i16", "mW"),
    0x1C: ("StateOfCharge", "u16", "%"),
    0x1E: ("InternalTemperature", "u16", "0.1K"),
    0x20: ("StateOfHealth", "soh", "%"),
    0x28: ("RemainingCapacityUnfiltered", "u16", "mAh"),
    0x2A: ("RemainingCapacityFiltered", "u16", "mAh"),
    0x2C: ("FullChargeCapacityUnfiltered", "u16", "mAh"),
    0x2E: ("FullChargeCapacityFiltered", "u16", "mAh"),
    0x30: ("StateOfChargeUnfiltered", "u16", "%"),
}

FLAG_BITS = {   # Flags(), high byte bits 15-8, low byte bits 7-0
    15: "OT", 14: "UT", 9: "FC", 8: "CHG",
    7: "OCVTAKEN", 6: "ITPOR", 5: "CFGUPMODE", 4: "BAT_DET", 3: "SOC1", 2: "SOCF", 1: "DSG", 0: "CORRECT",
}
SOH_STATUS = {0: "invalid", 1: "instant", 2: "initial (default Qmax)", 3: "learned Qmax"}


def decode_burst(data: bytes, start: int = BURST_START) -> dict:
    """Decode a block of consecutive registers starting at `start` (little-endian words)."""
    out = {}
    for code, (name, kind, unit) in STANDARD.items():
        o = code - start
        if o < 0 or o + 1 >= len(data):
            continue
        lo, hi = data[o], data[o + 1]
        w = lo | hi << 8
        out[f"raw_0x{code:02x}"] = w
        if kind == "flags":
            out[name] = [n for b, n in sorted(FLAG_BITS.items(), reverse=True) if w >> b & 1]
            continue
        if kind == "soh":
            out[name] = lo                               # percent
            out["StateOfHealthStatus"] = SOH_STATUS.get(hi, f"0x{hi:02x}")
            continue
        x = w - 0x10000 if kind == "i16" and w & 0x8000 else w
        if unit == "0.1K":
            out[name] = round(x / 10.0 - 273.15, 2)
        elif unit in ("mV", "mAh", "mA", "mW"):
            out[name] = x / 1000.0                       # V, Ah, A, W
        else:
            out[name] = x
    return out


class Bq27426Reader:
    """One rate-limited burst read per sample; keys shared with the SBS reader for the estimators:
    Voltage (V), Current (A, positive = discharge), Temperature (degC), RelativeStateOfCharge (%)."""

    def __init__(self, transport, addr: int = ADDR, min_interval_s: float = MIN_INTERVAL_S, clock=time):
        self.t, self.addr, self.min_interval_s, self.clock = transport, addr, min_interval_s, clock
        self.stats = ReadStats()
        self.pec = False
        self.sign_mismatch = 0
        self.implausible = 0
        self._last = None

    def _wait(self):
        if self._last is not None and self.min_interval_s > 0:
            dt = self.clock.monotonic() - self._last
            if dt < self.min_interval_s:
                self.clock.sleep(self.min_interval_s - dt)
        self._last = self.clock.monotonic()

    def burst(self) -> bytes:
        self._wait()
        return self.t.read(self.addr, BURST_START, BURST_END - BURST_START + 1)

    def sample(self, **_ignored) -> dict:
        """No retry within the same sample: a retry would be a third command in that second."""
        t0 = time.perf_counter()
        try:
            d = decode_burst(self.burst())
            if not (0.0 < d["Voltage"] < 6.0 and 0 <= d["StateOfCharge"] <= 100 and -60 < d["Temperature"] < 120):
                self.implausible += 1
                raise OSError("implausible BQ27426 reading")
            self.stats.ok += 1
        except OSError:
            self.stats.failed += 1
            keys = ("Voltage", "Current", "Temperature", "RelativeStateOfCharge")
            return {**{k: None for k in keys}, "t_read_ms": (time.perf_counter() - t0) * 1000}
        d["Current"] = -d["AverageCurrent"]                  # positive = discharge
        d["RelativeStateOfCharge"] = d["StateOfCharge"]
        d["Power"] = -d["AveragePower"]
        if "DSG" in d["Flags"] and d["Current"] < -0.05:     # gauge says discharging, current says charging
            self.sign_mismatch += 1
        d["t_read_ms"] = (time.perf_counter() - t0) * 1000
        return d

    def info(self) -> dict:
        return {"device": "TI BQ27426 (system-side Impedance Track)", "addr": f"0x{self.addr:02x}", "pec": False}

    def try_word(self, cmd: int, retries: int = 0):
        return None                                          # not used: single cell, no extra registers


class EmulatedBq27426:
    """The BQ27426 register interface on top of the emulated cell of EmulatedSmartBattery.

    Emulates the interface (address, register map, incremental reads, NACK above
    0x6B), not Impedance Track: its SoC is the same OCV start plus coulomb counting.
    """

    def __init__(self, cycle: str = "25degC_UDDS", seed: int = 0, error_rate: float = 0.0):
        from .smart_battery_emulator import EmulatedSmartBattery

        self.cell = EmulatedSmartBattery(cycle, pec=False, error_rate=error_rate, seed=seed)
        self.transfers_this_step = 0
        self.max_transfers_per_step = 0

    def __getattr__(self, name):                             # t, ref, k, n, done, cycle
        return getattr(self.cell, name)

    def advance(self, dt: float = 1.0):
        self.max_transfers_per_step = max(self.max_transfers_per_step, self.transfers_this_step)
        self.transfers_this_step = 0
        self.cell.advance(dt)

    def memory(self) -> bytes:
        c, k = self.cell, self.cell.k
        rem = max(c.rem_mah, 0.0)
        flags = 1 << 4                                                          # BAT_DET
        flags |= (1 << 1) if c.i_ma < 0 else 0                                  # DSG
        flags |= (1 << 9) if c.rsoc >= 100 else 0                               # FC
        words = {
            0x02: (c.T[k] + 273.15) * 10, 0x04: c.v[k] * 1000, 0x06: flags,
            0x08: rem, 0x0A: c.fcc_mah, 0x0C: rem, 0x0E: c.fcc_mah, 0x10: c.i_ma,
            0x18: c.i_ma * c.v[k], 0x1C: c.rsoc, 0x1E: (c.T[k] + 273.15) * 10,
            0x20: 100 | 2 << 8,                                                 # 100 %, initial Qmax
            0x28: rem, 0x2A: rem, 0x2C: c.fcc_mah, 0x2E: c.fcc_mah, 0x30: c.rsoc,
        }
        mem = bytearray(MAX_CODE + 1)
        for code, val in words.items():
            w = int(round(val)) & 0xFFFF
            mem[code], mem[code + 1] = w & 0xFF, w >> 8
        return bytes(mem)

    def read(self, addr: int, cmd: int, n: int) -> bytes:
        if addr != ADDR:
            raise OSError(121, "Remote I/O error (no device at this address)")
        if cmd > MAX_CODE or cmd + n - 1 > MAX_CODE:
            raise OSError(121, "Remote I/O error (address above 0x6B is NACKed)")
        if self.cell.done:
            raise OSError(121, "Remote I/O error (end of recorded cycle)")
        if self.cell.error_rate and self.cell.rng.random() < self.cell.error_rate:
            raise OSError(121, "Remote I/O error (emulated NACK)")
        self.transfers_this_step += 1
        return self.memory()[cmd:cmd + n]

    def close(self):
        pass


def consistent(log) -> dict:
    """Quick checks on a logged run: does the current sign agree with the DSG flag, and is SoC monotonic in discharge?"""
    cur = np.asarray(log["Current"], float)
    dsg = np.asarray(["DSG" in str(f) for f in log["Flags"]])
    strong = np.abs(cur) > 0.05
    agree = float(np.mean((cur[strong] > 0) == dsg[strong])) if strong.any() else float("nan")
    return {"current_sign_agrees_with_DSG": agree}
