import numpy as np
import pytest

from socest.coulomb import coulomb_count
from socest.ecm import ECMParams, TheveninModel
from socest.ocv import OCVCurve

from conftest import CAPACITY_AH, OCV_BP, PARAMS_2RC, SOC_BP


def test_ocv_hits_breakpoints_and_is_monotone(ocv):
    np.testing.assert_allclose(ocv(SOC_BP), OCV_BP, atol=2e-4)
    dense = ocv(np.linspace(0, 1, 500))
    assert np.all(np.diff(dense) >= 0)
    assert np.all(ocv.dsoc(np.linspace(0.01, 0.99, 50)) > 0)


def test_ocv_temperature_interpolation():
    curve = OCVCurve(SOC_BP, np.vstack([OCV_BP - 0.02, OCV_BP]), temps=[0.0, 20.0])
    mid = curve(0.5, temp=10.0)
    np.testing.assert_allclose(mid, 0.5 * (curve(0.5, 0.0) + curve(0.5, 20.0)))
    # held at the edge outside the temperature range
    np.testing.assert_allclose(curve(0.5, -30.0), curve(0.5, 0.0))


def test_rest_voltage_equals_ocv(model, ocv):
    v, _ = model.simulate(np.zeros(100), 1.0, 25.0, soc0=0.6)
    np.testing.assert_allclose(v, ocv(0.6), atol=1e-12)


def test_step_response_matches_analytic(model, ocv):
    """Constant discharge current I from rest: closed-form solution for each RC branch."""
    I, dt, n = 3.0, 1.0, 600
    v, states = model.simulate(np.full(n, I), dt, 25.0, soc0=0.8)
    t = np.arange(n) * dt
    p = PARAMS_2RC
    v_rc = sum(r * I * (1 - np.exp(-t / tau)) for r, tau in zip(p.r, p.tau))
    soc = 0.8 - I * t / (3600 * CAPACITY_AH)
    np.testing.assert_allclose(states[:, 0], soc, atol=1e-12)
    np.testing.assert_allclose(v, ocv(soc) - v_rc - p.r0 * I, atol=1e-10)


def test_discharge_lowers_voltage_and_soc(model):
    v_dis, s_dis = model.simulate(np.full(10, 2.0), 1.0, 25.0, soc0=0.5)
    v_chg, s_chg = model.simulate(np.full(10, -2.0), 1.0, 25.0, soc0=0.5)
    assert s_dis[-1, 0] < 0.5 < s_chg[-1, 0]
    assert v_dis[-1] < v_chg[-1]


def test_jacobians_match_finite_differences(model):
    x = np.array([0.43, 0.012, -0.004])
    i, dt, temp, h = 2.5, 0.1, 25.0, 1e-6
    x_next, A = model.transition(x, i, dt, temp)
    _, H = model.output(x, i, temp)
    for j in range(model.n_states):
        e = np.zeros(model.n_states)
        e[j] = h
        num_A = (model.transition(x + e, i, dt, temp)[0] - model.transition(x - e, i, dt, temp)[0]) / (2 * h)
        num_H = (model.output(x + e, i, temp)[0] - model.output(x - e, i, temp)[0]) / (2 * h)
        np.testing.assert_allclose(A[:, j], num_A, atol=1e-6)
        np.testing.assert_allclose(H[j], num_H, rtol=1e-3, atol=1e-4)


def test_coulomb_count_matches_model_soc(model, pulse_profile):
    _, states = model.simulate(pulse_profile, 1.0, 25.0, soc0=0.9)
    soc_cc = coulomb_count(pulse_profile, 1.0, 0.9, CAPACITY_AH)
    np.testing.assert_allclose(soc_cc, states[:, 0], atol=1e-12)


def test_coulomb_count_full_capacity():
    # 3 A for 1 h on a 3 Ah cell removes exactly 100 % SoC
    soc = coulomb_count(np.full(3601, 3.0), 1.0, 1.0, 3.0)
    np.testing.assert_allclose(soc[-1], 0.0, atol=1e-12)


def test_one_rc_model_is_supported(ocv):
    m = TheveninModel(ocv, ECMParams(r0=0.02, r=(0.01,), tau=(30.0,)), CAPACITY_AH)
    assert m.n_states == 2
    v, s = m.simulate(np.full(5, 1.0), 1.0, 25.0, soc0=0.5)
    assert s.shape == (5, 2) and np.all(np.isfinite(v))


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        ECMParams(r0=0.01, r=(0.01, 0.02), tau=(10.0,))


def test_butler_volmer_term():
    from socest.ecm import FARADAY, GAS_R, bv_overpotential

    assert bv_overpotential(2.0, 25.0, float("inf")) == 0.0
    assert bv_overpotential(0.0, -20.0, 0.2) == 0.0
    # odd in current, and linear with slope R T / (F i0) for small currents
    np.testing.assert_allclose(bv_overpotential(-1.5, -20.0, 0.2), -bv_overpotential(1.5, -20.0, 0.2))
    small = 1e-4
    np.testing.assert_allclose(bv_overpotential(small, 0.0, 0.5) / small, GAS_R * 273.15 / (FARADAY * 0.5), rtol=1e-6)
    # resistance falls with current
    assert bv_overpotential(6.0, -20.0, 0.2) / 6.0 < bv_overpotential(0.75, -20.0, 0.2) / 0.75


def test_model_output_includes_butler_volmer(ocv):
    from socest.ecm import bv_overpotential

    lin = TheveninModel(ocv, ECMParams(0.02, (0.01,), (30.0,)), CAPACITY_AH)
    bv = TheveninModel(ocv, ECMParams(0.02, (0.01,), (30.0,), i0=0.3), CAPACITY_AH)
    x = np.array([0.5, 0.004])
    np.testing.assert_allclose(lin.output(x, 2.0, -10.0)[0] - bv.output(x, 2.0, -10.0)[0],
                               bv_overpotential(2.0, -10.0, 0.3))
    np.testing.assert_allclose(lin.output(x, 2.0, -10.0)[1], bv.output(x, 2.0, -10.0)[1])  # same Jacobian


def test_effective_temperature():
    from socest.thermal import effective_temperature

    T = np.r_[np.full(40, -20.0), np.full(10, -16.0)]
    np.testing.assert_allclose(effective_temperature(T, 0.0), T)
    Te = effective_temperature(T, 1.0)
    np.testing.assert_allclose(Te[:40], -20.0)
    np.testing.assert_allclose(Te[40:], -12.0)
