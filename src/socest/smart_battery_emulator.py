"""Emulated smart battery: an SBS 1.1 fuel gauge on top of a recorded test cycle.

Lets the SMBus reader, the logger, the register identification and the edge
demo run end to end without a battery pack on the bench. It answers the
reader's byte-level transfers exactly as a gauge would: little-endian words,
length-prefixed blocks, an optional PEC byte, and a bus error (like a NACK) for
an unsupported command or a wrong address.

What it emulates, and what it does not:
    pack            one LG HG2 cell (1S1P); measurements are the recorded cycle
                    from demo/replay/, quantised to SBS resolution
                    (1 mV, 1 mA, 0.1 K, whole percent)
    fuel gauge      a deliberately simple one: SoC initialised from the OCV
                    curve at the first sample, then coulomb counting against
                    the cell's measured capacity. It stands in for the gauge's
                    RelativeStateOfCharge() so the pipeline has a "pack says"
                    signal; it is not a model of a commercial gauge (for
                    example TI Impedance Track), and its accuracy says nothing
                    about one.
    vendor area     a few words at manufacturer-specific codes, in the style of
                    real gauges: cell voltages at 0x3C-0x3F (CellVoltage4..1 as
                    on the TI BQ4050) and, at 0x30-0x33, emulator-defined
                    registers (temperature in 0.1 degC, current in 10 mA units
                    with discharge positive, a discharged-charge counter in
                    mAh, a constant). Their meaning is not given to the
                    register identification, which has to find it from data.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import sbs
from .cell import capacity_fn, load_model
from .replay import load_replay

VENDOR_TRUTH = {
    0x30: "temperature, 0.1 degC, signed",
    0x31: "current, 10 mA per LSB, discharge positive, signed",
    0x32: "discharged charge counter, mAh",
    0x33: "constant",
    0x3C: "CellVoltage4 (0, no fourth cell)",
    0x3D: "CellVoltage3 (0)",
    0x3E: "CellVoltage2 (0)",
    0x3F: "CellVoltage1, mV",
}


def _u16(x) -> int:
    return int(round(x)) & 0xFFFF


class EmulatedSmartBattery:
    def __init__(self, cycle: str = "25degC_UDDS", pec: bool = True, error_rate: float = 0.0,
                 corrupt_rate: float = 0.0, seed: int = 0, avg_window_s: int = 60):
        u = load_replay(cycle)
        self.cycle = cycle
        self.t = u.time_s.to_numpy(float)
        self.v = u.voltage_v.to_numpy(float)
        self.i = u.current_a.to_numpy(float)        # positive = discharge
        self.T = u.temp_degc.to_numpy(float)
        self.ref = u.soc_ref.to_numpy(float)
        self.n = len(self.t)
        self.k = 0
        self.pec, self.error_rate, self.corrupt_rate = pec, error_rate, corrupt_rate
        self.rng = np.random.default_rng(seed)
        self.addr = sbs.SMART_BATTERY_ADDR

        self.fcc_mah = round(1000 * capacity_fn()(float(self.T[0])))
        ocv = load_model().ocv
        grid = np.linspace(0, 1, 2001)
        soc0 = float(np.interp(self.v[0], ocv(grid, float(self.T[0])), grid))
        self.rem_mah = soc0 * self.fcc_mah
        self.discharged_mah = 0.0
        self.avg = deque(maxlen=avg_window_s)
        self._update_derived()

    # ------------------------------------------------------------ time
    @property
    def done(self) -> bool:
        return self.k >= self.n

    def advance(self, dt: float = 1.0):
        """Move to the next recorded sample; the gauge integrates the current over the step."""
        i = self.i[self.k]
        self.rem_mah -= i * dt / 3.6
        self.discharged_mah += max(i, 0.0) * dt / 3.6
        self.k += 1
        if not self.done:
            self._update_derived()

    def _update_derived(self):
        k = self.k
        self.i_ma = -self.i[k] * 1000.0                 # SBS sign: positive = charge
        self.avg.append(self.i_ma)
        self.rsoc = int(np.clip(round(100 * self.rem_mah / self.fcc_mah), 0, 100))

    # ------------------------------------------------------------ registers
    def words(self) -> dict:
        k = self.k
        v_mv, i_ma, avg_ma = self.v[k] * 1000, self.i_ma, float(np.mean(self.avg))
        rem = max(self.rem_mah, 0.0)
        status = 1 << 7                                                     # INITIALIZED
        status |= (1 << 6) if i_ma < 0 else 0                                 # DISCHARGING
        status |= (1 << 5) if self.rsoc >= 100 else 0                         # FULLY_CHARGED
        status |= (1 << 4) if self.rsoc <= 0 else 0                           # FULLY_DISCHARGED
        status |= (1 << 11) if v_mv < 2500 else 0                             # TERMINATE_DISCHARGE_ALARM
        rtte = 0xFFFF if avg_ma >= 0 else min(0xFFFE, int(rem / -avg_ma * 60))
        w = {
            0x00: 0, 0x01: 300, 0x02: 10, 0x03: 0x0000, 0x07: 1,
            0x08: _u16((self.T[k] + 273.15) * 10), 0x09: _u16(v_mv), 0x0A: _u16(i_ma), 0x0B: _u16(avg_ma),
            0x0C: 1, 0x0D: self.rsoc, 0x0E: int(np.clip(round(100 * rem / 3000), 0, 100)),
            0x0F: _u16(rem), 0x10: self.fcc_mah, 0x11: rtte, 0x12: rtte, 0x13: 0xFFFF,
            0x14: 1500, 0x15: 4200, 0x16: status, 0x17: 0, 0x18: 3000, 0x19: 3600,
            0x1A: 0x0031 if self.pec else 0x0021,          # SBS 1.1 (with PEC), no scaling
            0x1B: (2026 - 1980) * 512 + 11 * 32 + 21, 0x1C: 1,
            # vendor area (see module docstring)
            0x30: _u16(self.T[k] * 10), 0x31: _u16(-i_ma / 10), 0x32: _u16(self.discharged_mah),
            0x33: 0x5A17, 0x3C: 0, 0x3D: 0, 0x3E: 0, 0x3F: _u16(v_mv),
        }
        return w

    BLOCKS = {0x20: b"socest", 0x21: b"Emulated HG2 1S", 0x22: b"LION", 0x23: b"\x00" * 14}

    # ------------------------------------------------------------ transport
    def read(self, addr: int, cmd: int, n: int) -> bytes:
        if addr != self.addr:
            raise OSError(121, "Remote I/O error (no device at this address)")
        if self.error_rate and self.rng.random() < self.error_rate:
            raise OSError(121, "Remote I/O error (emulated NACK)")
        if self.done:
            raise OSError(121, "Remote I/O error (end of recorded cycle)")
        if cmd in self.BLOCKS:
            data = self.BLOCKS[cmd]
            payload = bytes([len(data)]) + data
        else:
            w = self.words().get(cmd)
            if w is None:
                raise OSError(121, f"Remote I/O error (command 0x{cmd:02x} not supported)")
            payload = bytes([w & 0xFF, w >> 8])
        if self.pec:
            payload += bytes([sbs.read_pec(addr, cmd, payload)])
        out = bytearray((payload + b"\xff" * n)[:n])
        if self.corrupt_rate and self.rng.random() < self.corrupt_rate:
            out[0] ^= 0x01
        return bytes(out)

    def close(self):
        pass
