import numpy as np

from socest.coulomb import coulomb_count
from socest.kalman import EKF

from conftest import CAPACITY_AH


def make_ekf(model, soc0, p0_soc=0.1**2, r=1e-4):
    x0 = np.array([soc0, 0.0, 0.0])
    P0 = np.diag([p0_soc, 1e-4, 1e-4])
    Q = np.diag([1e-10, 1e-8, 1e-8])
    return EKF(model, x0, P0, Q, r)


def test_update_moves_soc_toward_measurement(model, ocv):
    ekf = make_ekf(model, soc0=0.5)
    v_true = float(ocv(0.6))  # at rest, terminal voltage is OCV
    p_before = ekf.P[0, 0]
    innov = ekf.update(v_true, 0.0, 25.0)
    assert innov > 0
    # One linearised update may overshoot slightly (local OCV slope differs from
    # the secant), but it must cut the SoC error substantially.
    assert ekf.x[0] > 0.5
    assert abs(ekf.x[0] - 0.6) < 0.2 * abs(0.5 - 0.6)
    assert ekf.P[0, 0] < p_before


def test_covariance_stays_symmetric_positive_definite(model, pulse_profile):
    v, _ = model.simulate(pulse_profile, 1.0, 25.0, soc0=0.8)
    ekf = make_ekf(model, soc0=0.8)
    ekf.run(v, pulse_profile, 1.0, 25.0)
    np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-15)
    assert np.all(np.linalg.eigvalsh(ekf.P) > 0)


def test_ekf_recovers_from_wrong_initial_soc(model, pulse_profile):
    """Noisy synthetic data, initial SoC off by 20 points. Coulomb counting keeps the
    full offset; the EKF converges."""
    rng = np.random.default_rng(1)
    soc_true0 = 0.85
    v, states = model.simulate(pulse_profile, 1.0, 25.0, soc0=soc_true0)
    v_noisy = v + rng.normal(0.0, 0.005, v.size)

    ekf = make_ekf(model, soc0=soc_true0 - 0.20, r=0.005**2)
    out = ekf.run(v_noisy, pulse_profile, 1.0, 25.0)
    err_ekf = np.abs(out["soc"] - states[:, 0])
    err_cc = np.abs(coulomb_count(pulse_profile, 1.0, soc_true0 - 0.20, CAPACITY_AH) - states[:, 0])

    tail = slice(len(err_ekf) // 2, None)
    assert err_cc[tail].mean() > 0.19
    assert err_ekf[tail].max() < 0.02


def test_ekf_matches_truth_with_perfect_init(model, pulse_profile):
    v, states = model.simulate(pulse_profile, 1.0, 25.0, soc0=0.7)
    ekf = make_ekf(model, soc0=0.7, p0_soc=1e-6)
    out = ekf.run(v, pulse_profile, 1.0, 25.0)
    assert np.max(np.abs(out["soc"] - states[:, 0])) < 1e-3
