import numpy as np
import pytest

from socest.ecm import ECMParams, TheveninModel
from socest.pack import Adjacent, CellToPack, Passive, PackModel, simulate_drive, simulate_rest

from conftest import CAPACITY_AH, PARAMS_2RC


@pytest.fixture
def cell(ocv):
    return TheveninModel(ocv, PARAMS_2RC, CAPACITY_AH)


def make_pack(cell, n=4, p=1, cap=None, res=None):
    cap = np.ones((p, n)) if cap is None else np.asarray(cap, float).reshape(p, n)
    res = np.ones((p, n)) if res is None else np.asarray(res, float).reshape(p, n)
    return PackModel(cell, 25.0, cap, res)


def test_single_cell_matches_thevenin_model(cell):
    rng = np.random.default_rng(0)
    i = np.repeat(rng.uniform(-2, 5, 40), 15)
    v_ref, s_ref = cell.simulate(i, 1.0, 25.0, soc0=0.8)
    pack = make_pack(cell, n=1)
    pack.reset(np.array([[0.8]]))
    v = np.array([pack.step(np.array([[x]]), 1.0)[0, 0] for x in i])
    np.testing.assert_allclose(v, v_ref, atol=2e-4)
    np.testing.assert_allclose(pack.soc[0, 0], s_ref[-1, 0] - i[-1] / (3600 * CAPACITY_AH), atol=1e-12)


def test_capacity_and_resistance_scaling(cell):
    pack = make_pack(cell, n=2, cap=[1.0, 0.5], res=[1.0, 2.0])
    pack.reset(np.full((1, 2), 0.8))
    v = pack.step(np.full((1, 2), 3.0), 60.0)
    drop = 0.8 - pack.soc[0]
    np.testing.assert_allclose(drop[1], 2 * drop[0])          # half the capacity, twice the SoC drop
    assert v[0, 1] < v[0, 0]                                   # higher resistance, lower voltage


def test_stored_energy_matches_ocv_integral(cell):
    pack = make_pack(cell, n=1)
    pack.reset(np.array([[1.0]]))
    g = np.linspace(0, 1, 20001)
    expected = CAPACITY_AH * np.trapezoid(cell.ocv(g, 25.0), g)
    np.testing.assert_allclose(pack.stored_energy_wh()[0, 0], expected, rtol=1e-4)


@pytest.mark.parametrize("bal", [Adjacent(i_bal=0.3, eta=0.8), CellToPack(i_bal=0.3, eta=0.8)])
def test_converter_power_balance(bal):
    soc = np.array([[0.90, 0.85, 0.80, 0.86]])
    v = np.array([[4.05, 4.00, 3.95, 4.01]])
    i, loss = bal.currents(soc, v, np.array([True]), 0.0)
    # power taken from the cells = power dissipated in the converters
    np.testing.assert_allclose((i * v).sum(), loss[0], rtol=1e-12)
    assert loss[0] > 0


def test_passive_bleeds_only_cells_above_minimum():
    soc = np.array([[0.90, 0.80, 0.85]])
    v = np.array([[4.0, 3.9, 3.95]])
    i, loss = Passive(r_bleed=20.0).currents(soc, v, np.array([True]), 0.001)
    np.testing.assert_allclose(i, [[0.2, 0.0, 3.95 / 20]])
    np.testing.assert_allclose(loss, [4.0**2 / 20 + 3.95**2 / 20])


@pytest.mark.parametrize("bal", [Passive(), Adjacent(), CellToPack()])
def test_rest_balancing_converges_and_loses_energy(cell, bal):
    pack = make_pack(cell, n=4, cap=[1.0, 0.98, 1.02, 1.0])
    out = simulate_rest(pack, bal, np.array([[0.90, 0.86, 0.93, 0.88]]), dt=10.0)
    assert np.isfinite(out["t_balance_h"][0])
    assert out["final_spread"][0] < 0.005
    assert out["loss_wh"][0] > 0 and out["circuit_loss_wh"][0] > 0
    assert out["circuit_loss_wh"][0] <= out["loss_wh"][0] * 1.05      # the rest is ohmic loss inside the cells


def test_active_balancing_recovers_capacity_passive_does_not(cell):
    """Weak cell limits the discharge; active balancing moves charge into it."""
    cap = [1.0, 0.9, 1.0, 1.0]
    soc0 = np.full((1, 4), 0.95)
    prof = np.full(600, 1.5)
    res = {}
    for name, bal in (("none", None), ("passive", Passive()), ("c2p", CellToPack(eta=0.9))):
        pack = make_pack(cell, n=4, cap=cap)
        res[name] = simulate_drive(pack, bal, soc0, prof, dt=10.0, v_cut=3.0, max_repeats=50)["energy_wh"][0]
    assert res["c2p"] > res["none"] >= res["passive"]
