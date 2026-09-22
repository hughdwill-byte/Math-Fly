"""Smoke tests: the whole pipeline runs and learns above chance on easy tasks.
Run with:  pytest -q   (or)   python -m pytest tests/
"""

import numpy as np

from mathfly.connectome import make_synthetic_connectome, load_connectome
from mathfly.io_encoding import IOEncoder, tokenize_problem, VOCAB
from mathfly.model import ReservoirModel
from mathfly.envs import CURRICULUM, CURRICULUM_BY_NAME, make_dataset
from mathfly.train import train_reservoir


def test_connectome_dales_law():
    c = make_synthetic_connectome(n=300, seed=1)
    assert c.n == 300
    # Dale's law: every neuron's outgoing edges share one sign.
    W = c.W.tocsc()
    for j in range(c.n):
        col = W.getcol(j).data
        if len(col):
            assert np.all(col >= 0) or np.all(col <= 0)


def test_spectral_rescale():
    c = make_synthetic_connectome(n=400, seed=2).rescale_spectral_radius(1.0)
    assert abs(c.spectral_radius() - 1.0) < 0.2


def test_encode_shapes():
    enc = IOEncoder(n_neurons=200, n_input=32, n_readout=64, n_answers=20)
    U = enc.encode(tokenize_problem(3, "+", 4))
    assert U.shape[1] == 200
    assert U.shape[0] > 0


def test_forward_runs():
    c = make_synthetic_connectome(n=200, seed=3).rescale_spectral_radius(1.1)
    enc = IOEncoder(c.n, n_input=32, n_readout=64, n_answers=19)
    m = ReservoirModel(c, enc)
    f = m.features(tokenize_problem(2, "+", 3))
    assert np.all(np.isfinite(f))


def test_learns_above_chance():
    """Reservoir should beat chance on single-digit addition after training."""
    config = {"synthetic_n": 500, "synthetic_density": 0.03, "spectral_radius": 1.1,
              "n_readout": 200, "n_train": 250, "n_eval": 150,
              "tasks": ["add_1digit"], "out_dir": "runs/_test", "seed": 0}
    _, results = train_reservoir(config)
    r = results[0]
    assert r["test_acc"] > 3 * r["chance"], r


def test_digit_encode_decode_roundtrip():
    enc = IOEncoder(n_neurons=100, n_digits=4)
    for a in [0, 7, 42, 100, 999, 1234]:
        digits = enc.target_digits(a)
        onehot = enc.target_digits_onehot(a)          # perfect "logits"
        assert enc.decode_digits(onehot) == min(a, 9999)
        assert digits[0] == a % 10                     # units-first


def test_digit_serial_learns():
    """Digit-serial readout should beat chance on units digit of addition."""
    from mathfly.train import train_reservoir_digits
    config = {"synthetic_n": 600, "synthetic_density": 0.03, "spectral_radius": 1.1,
              "n_readout": 256, "n_train": 500, "n_eval": 200, "n_digits": 2,
              "tasks": ["add_1digit"], "out_dir": "runs/_digtest", "seed": 0}
    _, results = train_reservoir_digits(config)
    # per-digit accuracy well above 10% chance
    assert results[0]["digit_acc"] > 0.25, results


def test_gym_env_roundtrip():
    from mathfly.gym_env import make_env
    env = make_env({"synthetic_n": 300}, task_name="add_1digit",
                   n_input=32, n_readout=64)
    obs, info = env.reset(seed=0)
    assert obs.shape == (300,)
    assert "answer" in info
    term = False
    steps = 0
    while not term and steps < 1000:
        obs, r, term, trunc, info = env.step(info["answer"])  # always answer correctly
        steps += 1
    assert term
    assert r == 1.0 and info["correct"]      # correct action -> reward 1
