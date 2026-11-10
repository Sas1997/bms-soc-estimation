"""Streaming estimators must reproduce the batch estimators sample for sample."""

import numpy as np
import pytest

from socest.cell import CHAR
from socest.coulomb import coulomb_count
from socest.streaming import StreamingCoulomb


def test_streaming_coulomb_matches_batch():
    rng = np.random.default_rng(0)
    i = rng.normal(1.0, 2.0, 500)
    s = StreamingCoulomb(2.8, soc0=0.9)
    np.testing.assert_allclose([s.step(0.0, x, 25.0) for x in i], coulomb_count(i, 1.0, 0.9, 2.8), atol=1e-12)


needs_model = pytest.mark.skipif(not (CHAR / "selected.json").exists() or not (CHAR.parents[1] / "results" / "m2_ecm_selection.csv").exists(),
                                 reason="identified cell model not present (run the pipeline first)")


def _signals(n=400):
    rng = np.random.default_rng(1)
    i = np.repeat(rng.uniform(-1.0, 5.0, n // 20), 20)
    v = 3.9 - 0.03 * i - 0.0005 * np.arange(n) + rng.normal(0, 0.002, n)
    T = 24.0 + 0.004 * np.arange(n)
    return v, i, T


@needs_model
def test_streaming_ekf_matches_batch():
    from socest.estimators import EKFEstimator
    from socest.streaming import StreamingEKF

    v, i, T = _signals()
    est = EKFEstimator(k_r=1.0, q_soc=1e-5, q_v=1e-3)
    batch = est(v, i, T, 0.9)
    # the batch thermal model takes the mean of the first 30 samples as ambient
    s = StreamingEKF(est, temp0=float(np.mean(T[:30])), soc0=0.9)
    s.model.capacity_ah = est.cap_of(T[0])
    stream = np.array([s.step(a, b, c) for a, b, c in zip(v, i, T)])
    np.testing.assert_allclose(stream, batch, atol=1e-9)


def test_streaming_nn_matches_batch(tmp_path, model):
    torch = pytest.importorskip("torch")
    from socest.ml.data import make_cycle
    from socest.ml.train import train
    import socest.estimators as E
    from socest.streaming import StreamingNN

    rng = np.random.default_rng(2)
    cyc = []
    for _ in range(2):
        i = np.repeat(rng.uniform(-1.0, 5.0, 30), 20)
        v, st = model.simulate(i, 1.0, 25.0, soc0=1.0)
        cyc.append(make_cycle(v, i, np.full(i.size, 25.0), st[:, 0]))
    train(cyc[:1], cyc[1:], tmp_path / "m", model_name="transformer", model_kwargs={"patch": 5}, window=40, stride=10,
          max_epochs=1, batch_size=32, seed=0, device="cpu")
    old = E.MODELS
    E.MODELS = tmp_path
    try:
        est = E.NeuralEstimator("m")
    finally:
        E.MODELS = old
    x = cyc[1]["x"]
    batch = est(x[:, 0], x[:, 1], x[:, 2])
    s = StreamingNN(est)
    stream = np.array([s.step(*row) for row in x[:120]])
    np.testing.assert_allclose(stream, batch[:120], atol=1e-5)
