"""Smart Battery Data Specification (SBS) 1.1: register map and decoding.

A smart battery pack has a fuel-gauge chip that answers numbered commands on
the SMBus (a two-wire bus based on I2C). The meaning of commands 0x00-0x23 is
fixed by the specification; codes from 0x24 up are reserved or
manufacturer-specific and differ between gauge families.

Source: Smart Battery Data Specification, Revision 1.1, SBS Implementers Forum,
11 December 1998 (sections 5.1-5.4 and Appendices A and C). The battery answers
at 7-bit SMBus address 0x0B (0x16 as an 8-bit write address).

Units and conventions, as the specification defines them:
    Voltage()         mV, multiplied by 10^VScale (SpecificationInfo bits 8-11)
    Current()         mA, signed, positive = charge, multiplied by 10^IPScale (bits 12-15)
    capacities        mAh, or 10 mWh when BatteryMode() bit 15 (CAPACITY_MODE) is set
    Temperature()     0.1 K
    times             minutes; 65535 means "not applicable" (for example not discharging)

This package uses positive = discharge for current (as in the rest of socest),
so decoded currents are negated.
"""

from __future__ import annotations

from dataclasses import dataclass

SMART_BATTERY_ADDR = 0x0B


@dataclass(frozen=True)
class Command:
    code: int
    name: str
    kind: str          # "u16", "i16", "flags", "date", "block"
    unit: str          # "mV", "mA", "cap", "0.1K", "%", "min", "count", "" ; "cap" = mAh or 10 mWh
    scale: str = ""    # "V" (VScale), "IP" (IPScale) or ""


# Read commands of SBS 1.1. Read/write commands (0x00-0x04) are only ever read here.
COMMANDS = {c.code: c for c in [
    Command(0x00, "ManufacturerAccess", "u16", ""),
    Command(0x01, "RemainingCapacityAlarm", "u16", "cap", "IP"),
    Command(0x02, "RemainingTimeAlarm", "u16", "min"),
    Command(0x03, "BatteryMode", "flags", ""),
    Command(0x04, "AtRate", "i16", "mA", "IP"),
    Command(0x05, "AtRateTimeToFull", "u16", "min"),
    Command(0x06, "AtRateTimeToEmpty", "u16", "min"),
    Command(0x07, "AtRateOK", "u16", ""),
    Command(0x08, "Temperature", "u16", "0.1K"),
    Command(0x09, "Voltage", "u16", "mV", "V"),
    Command(0x0A, "Current", "i16", "mA", "IP"),
    Command(0x0B, "AverageCurrent", "i16", "mA", "IP"),
    Command(0x0C, "MaxError", "u16", "%"),
    Command(0x0D, "RelativeStateOfCharge", "u16", "%"),
    Command(0x0E, "AbsoluteStateOfCharge", "u16", "%"),
    Command(0x0F, "RemainingCapacity", "u16", "cap", "IP"),
    Command(0x10, "FullChargeCapacity", "u16", "cap", "IP"),
    Command(0x11, "RunTimeToEmpty", "u16", "min"),
    Command(0x12, "AverageTimeToEmpty", "u16", "min"),
    Command(0x13, "AverageTimeToFull", "u16", "min"),
    Command(0x14, "ChargingCurrent", "u16", "mA"),       # excluded from scaling (SpecificationInfo)
    Command(0x15, "ChargingVoltage", "u16", "mV"),       # excluded from scaling (SpecificationInfo)
    Command(0x16, "BatteryStatus", "flags", ""),
    Command(0x17, "CycleCount", "u16", "count"),
    Command(0x18, "DesignCapacity", "u16", "cap", "IP"),
    Command(0x19, "DesignVoltage", "u16", "mV", "V"),
    Command(0x1A, "SpecificationInfo", "flags", ""),
    Command(0x1B, "ManufactureDate", "date", ""),
    Command(0x1C, "SerialNumber", "u16", ""),
    Command(0x20, "ManufacturerName", "block", ""),
    Command(0x21, "DeviceName", "block", ""),
    Command(0x22, "DeviceChemistry", "block", ""),
    Command(0x23, "ManufacturerData", "block", ""),
]}
BY_NAME = {c.name: c for c in COMMANDS.values()}

# Read every second by the logger and the edge demo.
DYNAMIC = ["Voltage", "Current", "AverageCurrent", "Temperature", "RelativeStateOfCharge",
           "AbsoluteStateOfCharge", "RemainingCapacity", "FullChargeCapacity", "RunTimeToEmpty",
           "BatteryStatus", "MaxError"]
# Read once at start.
STATIC = ["BatteryMode", "SpecificationInfo", "DesignCapacity", "DesignVoltage", "CycleCount",
          "ManufactureDate", "SerialNumber", "ManufacturerName", "DeviceName", "DeviceChemistry"]

BATTERY_STATUS_BITS = {
    15: "OVER_CHARGED_ALARM", 14: "TERMINATE_CHARGE_ALARM", 12: "OVER_TEMP_ALARM",
    11: "TERMINATE_DISCHARGE_ALARM", 9: "REMAINING_CAPACITY_ALARM", 8: "REMAINING_TIME_ALARM",
    7: "INITIALIZED", 6: "DISCHARGING", 5: "FULLY_CHARGED", 4: "FULLY_DISCHARGED",
}
ERROR_CODES = {0: "OK", 1: "Busy", 2: "ReservedCommand", 3: "UnsupportedCommand", 4: "AccessDenied",
               5: "Overflow/Underflow", 6: "BadSize", 7: "UnknownError"}
BATTERY_MODE_BITS = {
    15: "CAPACITY_MODE", 14: "CHARGER_MODE", 13: "ALARM_MODE", 9: "PRIMARY_BATTERY",
    8: "CHARGE_CONTROLLER_ENABLED", 7: "CONDITION_FLAG", 1: "PRIMARY_BATTERY_SUPPORT",
    0: "INTERNAL_CHARGE_CONTROLLER",
}


# ---------------------------------------------------------------- PEC
def crc8(data: bytes | list[int], crc: int = 0) -> int:
    """SMBus packet error code: CRC-8, polynomial x^8 + x^2 + x + 1 (0x07), initial value 0."""
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def read_pec(addr: int, cmd: int, payload: bytes) -> int:
    """PEC of a Read Word / Read Block transaction: covers both address bytes, the command and the data."""
    return crc8([(addr << 1) | 0, cmd, (addr << 1) | 1, *payload])


# ---------------------------------------------------------------- decoding
def to_i16(w: int) -> int:
    return w - 0x10000 if w & 0x8000 else w


def flags(word: int, table: dict) -> list[str]:
    return [name for bit, name in sorted(table.items(), reverse=True) if word >> bit & 1]


def battery_status(word: int) -> dict:
    return {"flags": flags(word, BATTERY_STATUS_BITS), "error": ERROR_CODES.get(word & 0x0F, f"0x{word & 0x0F:x}")}


def specification_info(word: int) -> dict:
    version = (word >> 4) & 0x0F
    return {"revision": word & 0x0F, "version": {1: "1.0", 2: "1.1", 3: "1.1 with PEC"}.get(version, str(version)),
            "pec": version == 3, "vscale": (word >> 8) & 0x0F, "ipscale": (word >> 12) & 0x0F}


def manufacture_date(word: int) -> str:
    """(year - 1980) * 512 + month * 32 + day."""
    return f"{1980 + (word >> 9):04d}-{(word >> 5) & 0x0F:02d}-{word & 0x1F:02d}"


@dataclass
class Scaling:
    """Pack-level scale factors read once from SpecificationInfo() and BatteryMode()."""

    vscale: int = 0
    ipscale: int = 0
    capacity_mwh: bool = False   # BatteryMode CAPACITY_MODE: capacities in 10 mWh instead of mAh

    @classmethod
    def from_words(cls, spec_info: int | None, battery_mode: int | None):
        s = specification_info(spec_info) if spec_info is not None else {"vscale": 0, "ipscale": 0}
        mwh = bool(battery_mode >> 15 & 1) if battery_mode is not None else False
        return cls(s["vscale"], s["ipscale"], mwh)


def decode(name: str, raw, scaling: Scaling = Scaling()):
    """Engineering value of one register.

    Returns volts, amperes (positive = discharge), degC, %, minutes (None for 65535),
    capacities in Ah (or Wh when the pack reports in 10 mWh), strings for blocks.
    """
    c = BY_NAME[name]
    if c.kind == "block":
        return bytes(raw).split(b"\x00")[0].decode("ascii", "replace")
    w = int(raw)
    if c.kind == "flags":
        if name == "BatteryStatus":
            return battery_status(w)
        if name == "SpecificationInfo":
            return specification_info(w)
        return flags(w, BATTERY_MODE_BITS)
    if c.kind == "date":
        return manufacture_date(w)
    x = to_i16(w) if c.kind == "i16" else w
    mult = 10 ** (scaling.vscale if c.scale == "V" else scaling.ipscale if c.scale == "IP" else 0)
    if c.unit == "mV":
        return x * mult / 1000.0
    if c.unit == "mA":
        v = x * mult / 1000.0
        return -v if name in ("Current", "AverageCurrent", "AtRate") else v
    if c.unit == "cap":
        return x * mult * (10.0 if scaling.capacity_mwh else 1.0) / 1000.0
    if c.unit == "0.1K":
        return round(x / 10.0 - 273.15, 2)
    if c.unit == "min":
        return None if w == 0xFFFF else x
    return x
