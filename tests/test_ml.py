import numpy as np
import pytest

torch = pytest.importorskip("torch")

from socest.ml.data import Normalizer, WindowDataset, make_cycle  # noqa: E402
from socest.ml.models import SoCLSTM, SoCTransformer, n_params  # noqa: E402
from socest.ml.train import load, predict, train  # noqa: E402

from conftest import PARAMS_2RC, CAPACITY_AH  # noqa: E402


def synthetic_cycles(model, n_cycles, seed):
    rng = np.random.default_rng(seed)
    out = []
    for c in range(n_cycles):
        i = np.repeat(rng.uniform(-1.0, 6.0, 120), 30)  # 1 Hz, 1 h
        v, s = model.simulate(i, 1.0, 25.0, soc0=1.0)
        out.append(make_cycle(v, i, np.full(i.size, 25.0), s[:, 0], name=f"syn{c}"))
    return out


def test_model_shapes_and_size():
    x = torch.randn(4, 120, 3)
    tr = SoCTransformer(window=120)
    ls = SoCLSTM()
    assert tr(x).shape == (4,) and ls(x).shape == (4,)
    assert n_params(tr) < 50_000 and n_params(ls) < 50_000


def test_windows_do_not_cross_cycles():
    a = make_cycle(np.ones(10), np.zeros(10), np.zeros(10), np.linspace(1, 0.9, 10))
    b = make_cycle(np.full(10, 5.0), np.zeros(10), np.zeros(10), np.linspace(0.5, 0.4, 10))
    norm = Normalizer(mean=np.zeros(3), std=np.ones(3))
    ds = WindowDataset([a, b], norm, window=4)
    assert len(ds) == 20
    x, y = ds[10]  # first step of cycle b: padded with b's own first sample only
    assert torch.all(x[:, 0] == 5.0) and y.item() == pytest.approx(0.5)


def test_normalizer_uses_only_given_cycles():
    a = make_cycle(np.full(5, 3.0), np.zeros(5), np.zeros(5), np.zeros(5))
    norm = Normalizer.fit([a])
    np.testing.assert_allclose(norm.mean, [3.0, 0.0, 0.0])


@pytest.mark.parametrize("name", ["lstm", "transformer"])
def test_training_learns_on_synthetic_data(tmp_path, model, name):
    cycles = synthetic_cycles(model, 4, seed=0)
    meta = train(cycles[:3], cycles[3:], tmp_path, model_name=name, window=60, stride=10,
                 max_epochs=6, patience=10, batch_size=64, seed=0, device="cpu")
    trivial = float(np.sqrt(np.mean((cycles[3]["y"] - cycles[3]["y"].mean()) ** 2)))
    assert meta["best_val_rmse"] < 0.5 * trivial
    m, norm, meta2 = load(tmp_path)
    pred = predict(m, cycles[3], norm, meta2["window"])
    assert pred.shape == cycles[3]["y"].shape
