"""Read-only access to a smart battery over SMBus.

The reader implements only the two SMBus read protocols the Smart Battery Data
Specification uses, Read Word and Read Block: the host sends the battery address
and one command byte, then reads the reply after a repeated start. There is no
write function in this module, so it cannot change gauge settings, seal or
unseal a pack, or switch its protection FETs.

Transports
    LinuxI2C           /dev/i2c-N through smbus2 (Raspberry Pi: bus 1 on GPIO 2/3)
    EmulatedSmartBattery (socest.smart_battery_emulator) for tests and the demo

Both expose read(addr, cmd, n) -> n bytes, so everything above them (PEC checks,
retries, decoding, logging, the estimators) runs the same code on real hardware
and in emulation.

Block reads are done as one combined transfer of 33 bytes (+1 for PEC) rather
than with the kernel's SMBus block-read call, which many I2C adapters, the
Raspberry Pi's included, do not support. The first byte is the length.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import sbs


class PECError(IOError):
    pass


class LinuxI2C:
    """Combined write-read transfers on a Linux I2C bus (needs the smbus2 package)."""

    def __init__(self, bus: int = 1):
        from smbus2 import SMBus, i2c_msg

        self._msg = i2c_msg
        self.bus = SMBus(bus)

    def read(self, addr: int, cmd: int, n: int) -> bytes:
        w, r = self._msg.write(addr, [cmd]), self._msg.read(addr, n)
        self.bus.i2c_rdwr(w, r)
        return bytes(r)

    def close(self):
        self.bus.close()


@dataclass
class ReadStats:
    ok: int = 0
    retried: int = 0
    failed: int = 0
    pec_errors: int = 0
    by_cmd_failed: dict = field(default_factory=dict)


class SmartBatteryReader:
    """Reads and decodes SBS registers; retries transient bus errors.

    pec: True, False or None (use PEC when SpecificationInfo() says the gauge supports it).
    """

    def __init__(self, transport, addr: int = sbs.SMART_BATTERY_ADDR, pec: bool | None = None,
                 retries: int = 3, retry_delay_s: float = 0.02):
        self.t, self.addr, self.retries, self.retry_delay_s = transport, addr, retries, retry_delay_s
        self.stats = ReadStats()
        if pec is None:
            # Without PEC a reply is unchecked, so read SpecificationInfo() until two reads agree.
            self.pec = False
            prev = None
            for _ in range(5):
                spec = self.read_word(0x1A)
                if spec == prev:
                    break
                prev = spec
            else:
                raise OSError("SpecificationInfo() reads disagree; check the bus")
            self.pec = sbs.specification_info(spec)["pec"]
        else:
            self.pec = bool(pec)
            spec = self.read_word(0x1A)
        self.spec_info = sbs.specification_info(spec)
        mode = self.try_word(0x03)
        self.scaling = sbs.Scaling.from_words(spec, mode)

    # ------------------------------------------------------------ raw
    def _retry(self, fn, cmd, retries=None):
        """retries=0 is used for scanned codes, which may not exist; those are not counted in stats."""
        count = retries != 0
        retries = self.retries if retries is None else retries
        for attempt in range(retries + 1):
            try:
                out = fn()
                self.stats.ok += count
                return out
            except (OSError, PECError) as e:
                if isinstance(e, PECError) and count:
                    self.stats.pec_errors += 1
                if attempt == retries:
                    if count:
                        self.stats.failed += 1
                        self.stats.by_cmd_failed[cmd] = self.stats.by_cmd_failed.get(cmd, 0) + 1
                    raise
                self.stats.retried += 1
                time.sleep(self.retry_delay_s)

    def read_word(self, cmd: int, retries: int | None = None) -> int:
        def once():
            raw = self.t.read(self.addr, cmd, 3 if self.pec else 2)
            if self.pec and raw[2] != sbs.read_pec(self.addr, cmd, raw[:2]):
                raise PECError(f"PEC mismatch on command 0x{cmd:02x}")
            return raw[0] | raw[1] << 8

        return self._retry(once, cmd, retries)

    def read_block(self, cmd: int) -> bytes:
        def once():
            raw = self.t.read(self.addr, cmd, 33 + (1 if self.pec else 0))
            n = raw[0]
            if n > 32:
                raise OSError(f"block length {n} > 32 on command 0x{cmd:02x}")
            if self.pec and raw[1 + n] != sbs.read_pec(self.addr, cmd, raw[:1 + n]):
                raise PECError(f"PEC mismatch on command 0x{cmd:02x}")
            return bytes(raw[1:1 + n])

        return self._retry(once, cmd)

    def try_word(self, cmd: int, retries: int | None = None):
        try:
            return self.read_word(cmd, retries)
        except (OSError, PECError):
            return None

    # ------------------------------------------------------------ decoded
    def read(self, name: str):
        c = sbs.BY_NAME[name]
        raw = self.read_block(c.code) if c.kind == "block" else self.read_word(c.code)
        return sbs.decode(name, raw, self.scaling), raw

    def info(self) -> dict:
        """Static registers, decoded; None where the gauge does not answer."""
        out = {}
        for name in sbs.STATIC:
            try:
                out[name] = self.read(name)[0]
            except (OSError, PECError):
                out[name] = None
        out["pec"] = self.pec
        return out

    def sample(self, names=sbs.DYNAMIC, scan=()) -> dict:
        """One reading of the dynamic registers plus raw words of the codes in `scan`.

        Keys: decoded values by register name, raw words as 'raw_0x..', and
        't_read_ms' (time spent on the bus for this sample).
        """
        t0 = time.perf_counter()
        out = {}
        for name in names:
            try:
                val, raw = self.read(name)
            except (OSError, PECError):
                val, raw = None, None
            out[name] = val
            out[f"raw_0x{sbs.BY_NAME[name].code:02x}"] = raw
        for cmd in scan:     # one attempt: an unknown code may simply not be implemented
            out[f"raw_0x{cmd:02x}"] = self.try_word(cmd, retries=0)
        out["t_read_ms"] = (time.perf_counter() - t0) * 1000
        return out


def cell_inputs(s: dict, n_series: int = 1, n_parallel: int = 1, cell_voltage: float | None = None):
    """Per-cell (voltage V, current A positive = discharge, temperature degC) from one pack sample.

    The estimators in this package model one LG HG2 cell. For a pack, the pack
    voltage is divided by the number of series cells (or a cell-voltage register
    is used), and the pack current by the number of parallel strings.
    """
    v = cell_voltage if cell_voltage is not None else (s["Voltage"] / n_series if s.get("Voltage") is not None else None)
    i = s["Current"] / n_parallel if s.get("Current") is not None else None
    return v, i, s.get("Temperature")
