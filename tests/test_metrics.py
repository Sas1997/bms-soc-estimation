import numpy as np
import pytest

from socest.metrics import Perturbation, convergence_time, soc_errors


def test_soc_errors_known_values():
    ref = np.zeros(4)
    est = np.array([0.1, -0.1, 0.1, -0.3])
    m = soc_errors(est, ref)
    assert m["mae"] == pytest.approx(0.15)
    assert m["max_abs"] == pytest.approx(0.3)
    assert m["rmse"] == pytest.approx(np.sqrt((0.01 * 3 + 0.09) / 4))
    assert soc_errors(est, ref, skip=3)["max_abs"] == pytest.approx(0.3)


def test_convergence_time():
    t = np.arange(10.0)
    err = np.array([0.2, 0.1, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert convergence_time(err, np.zeros(10), t, band=0.02) == 3.0
    assert convergence_time(np.full(10, 0.5), np.zeros(10), t) is None


def test_perturbation_bias_and_reproducible_noise():
    v, i, t = np.full(1000, 3.7), np.zeros(1000), np.full(1000, 25.0)
    p = Perturbation("bias", i_bias=0.1, v_noise_std=0.01)
    v1, i1, _ = p.apply(v, i, t, seed=5)
    v2, _, _ = p.apply(v, i, t, seed=5)
    np.testing.assert_allclose(i1, 0.1)
    np.testing.assert_array_equal(v1, v2)
    assert abs(np.std(v1 - 3.7) - 0.01) < 0.002
