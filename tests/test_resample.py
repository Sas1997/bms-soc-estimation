import numpy as np
import pandas as pd

from socest.resample import to_uniform


def test_charge_is_preserved_with_irregular_sampling():
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.05, 0.15, 2000))
    t -= t[0]
    i = rng.normal(2.0, 3.0, t.size)
    df = pd.DataFrame({"time_s": t, "current_a": i, "voltage_v": np.full(t.size, 3.7), "temp_degc": np.full(t.size, 25.0)})
    u = to_uniform(df, 1.0)
    n = len(u)
    raw_ah = np.sum(i[:-1] * np.diff(t)[: len(i) - 1] * (t[1:] <= n)) / 3600  # samples fully inside the bins
    # exact check on the integral up to the last full bin
    cut = n * 1.0
    seg = np.minimum(t[1:], cut) - np.minimum(t[:-1], cut)
    exact = np.sum(i[:-1] * seg) / 3600
    np.testing.assert_allclose(u.current_a.sum() * 1.0 / 3600, exact, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(u.voltage_v, 3.7)
    assert raw_ah != 0  # sanity that the profile is non-trivial


def test_constant_signal_and_soc_alignment():
    t = np.arange(0, 10.01, 0.1)
    df = pd.DataFrame({"time_s": t, "current_a": np.full(t.size, 1.5), "voltage_v": np.full(t.size, 4.0),
                       "temp_degc": np.full(t.size, 20.0), "soc_ref": 1.0 - t / 100})
    u = to_uniform(df, 1.0)
    assert len(u) == 10
    np.testing.assert_allclose(u.current_a, 1.5)
    np.testing.assert_allclose(u.soc_ref, 1.0 - np.arange(10) / 100)  # value at bin start
