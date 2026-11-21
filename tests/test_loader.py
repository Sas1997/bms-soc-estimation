"""Loader tests on synthetic .mat files that mimic the documented LG HG2 layout.
File names are copied from the public Mendeley file listing (25degC and n20degC)."""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

# Skipped where h5py is not installed (for example on the Raspberry Pi edge device).
h5py = pytest.importorskip("h5py")

from socest.datasets import lg_hg2  # noqa: E402

REAL_NAMES = [
    ("10-25-18_07.39 549_HPPC_25degC_LGHG2.mat", "549", "HPPC", "hppc"),
    ("10-26-18_09.29 549_C20DisCh_25degC_LGHG2.mat", "549", "C20DisCh", "ocv_c20"),
    ("10-28-18_01.48 549_Dis_0p5C_25degC_LGHG2.mat", "549", "Dis_0p5C", "discharge"),
    ("10-28-18_13.10 551_Cap_1C_25degC_LGHG2.mat", "551", "Cap_1C", "discharge"),
    ("10-28-18_17.44 551_UDDS_25degC_LGHG2.mat", "551", "UDDS", "drive_cycle"),
    ("10-30-18_06.27 552_Mixed3_25degC_LGHG2.mat", "552", "Mixed3", "drive_cycle"),
    ("10-31-18_15.54 552_PausCycl_25degC_LGHG2.mat", "552", "PausCycl", "pause_cycle"),
    ("10-28-18_11.04 551_Charge1.mat", "551", "Charge1", "charge"),
    ("12-24-1822.25 610_Charge5.mat", "610", "Charge5", "charge"),  # missing underscore in the original
    ("12-25-18_01.31 610_US06_n20degC_LGHG2.mat", "610", "US06", "drive_cycle"),
]


@pytest.mark.parametrize("name,series,test,kind", REAL_NAMES)
def test_parse_real_file_names(name, series, test, kind):
    tf = lg_hg2.parse_name(Path("raw/25degC") / name)
    assert (tf.series, tf.test, tf.kind) == (series, test, kind)
    assert tf.ambient_degc == 25.0


def _columns(n=1000, current=-3.0):
    """100 s at 0.1 s with voltage falling 4.1 -> 3.6 V (the cell is discharging)."""
    t = np.arange(n) * 0.1
    return {
        "Time": t, "TimeStamp": t, "Voltage": np.linspace(4.1, 3.6, n),
        "Current": np.full(n, current), "Ah": current * t / 3600, "Wh": 3.7 * current * t / 3600,
        "Power": 3.7 * np.full(n, current), "Battery_Temp_degC": np.full(n, 25.3),
    }


def test_read_classic_mat_with_nested_struct(tmp_path):
    p = tmp_path / "x.mat"
    savemat(p, {"meas": _columns()})
    cols = lg_hg2.read_mat(p)
    assert set(lg_hg2.REQUIRED) <= set(cols)
    np.testing.assert_allclose(cols["Current"], -3.0)


def test_read_hdf5_mat(tmp_path):
    p = tmp_path / "y.mat"
    with h5py.File(p, "w") as f:
        g = f.create_group("meas")
        for k, v in _columns().items():
            g[k] = v[None, :]
    cols = lg_hg2.read_mat(p)
    assert cols["Voltage"].shape == (1000,)


def _write_folder(root, sign):
    d = root / "25degC"
    d.mkdir(parents=True)
    savemat(d / "10-26-18_09.29 549_C20DisCh_25degC_LGHG2.mat", {"meas": _columns(current=sign * 0.15)})
    savemat(d / "10-28-18_07.09 549_Dis_2C_25degC_LGHG2.mat", {"meas": _columns(current=sign * 6.0)})
    savemat(d / "10-28-18_17.44 551_UDDS_25degC_LGHG2.mat", {"meas": _columns(current=sign * 2.0)})
    return d


@pytest.mark.parametrize("raw_sign", [-1, 1])
def test_sign_detection_and_conversion(tmp_path, raw_sign):
    _write_folder(tmp_path, raw_sign)
    files = lg_hg2.list_files(tmp_path)
    assert len(files) == 3
    sign = lg_hg2.detect_discharge_sign(files)
    assert sign == raw_sign
    df = lg_hg2.to_frame(files[0], sign)
    assert (df["current_a"] > 0).all()  # discharge is positive after conversion
    assert df["time_s"].iloc[0] == 0.0
    assert df.attrs["kind"] == "ocv_c20"


def test_sign_detection_rejects_inconsistent_files(tmp_path):
    d = _write_folder(tmp_path, -1)
    savemat(d / "10-28-18_01.48 549_Dis_0p5C_25degC_LGHG2.mat", {"meas": _columns(current=+1.5)})
    with pytest.raises(ValueError, match="inconsistent"):
        lg_hg2.detect_discharge_sign(lg_hg2.list_files(tmp_path))


def test_sign_from_first_segment_when_file_also_recharges(tmp_path):
    """Real C/20 files hold a discharge then a charge of similar length (about 50/50 samples)."""
    d = tmp_path / "25degC"
    d.mkdir()
    dis, chg = _columns(current=-0.15), _columns(current=+0.15)
    chg["Voltage"] = chg["Voltage"][::-1]
    both = {k: np.concatenate([dis[k], chg[k] + (dis[k][-1] + 0.1 if k == "Time" else 0)]) for k in dis}
    savemat(d / "10-26-18_09.29 549_C20DisCh_25degC_LGHG2.mat", {"meas": both})
    assert lg_hg2.detect_discharge_sign(lg_hg2.list_files(tmp_path)) == -1


def test_sign_detection_rejects_segment_that_raises_voltage(tmp_path):
    d = tmp_path / "25degC"
    d.mkdir()
    c = _columns(current=-3.0)
    c["Voltage"] = c["Voltage"][::-1]
    savemat(d / "10-28-18_07.09 549_Dis_2C_25degC_LGHG2.mat", {"meas": c})
    with pytest.raises(ValueError, match="does not discharge"):
        lg_hg2.detect_discharge_sign(lg_hg2.list_files(tmp_path))


def test_split_rules_have_no_overlap():
    from socest.split import assign
    assert assign("25degC", "Mixed1") == "train"
    assert assign("25degC", "Mixed7") == "val"
    assert assign("n20degC", "US06") == "test_unseen_cycle"
    assert assign("10degC", "Mixed1") == "test_unseen_temp"  # held-out temperature wins over cycle name
    with pytest.raises(ValueError):
        assign("25degC", "HPPC")
