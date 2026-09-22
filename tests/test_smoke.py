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


def test_web_export_matches_model():
    """The browser forward pass (reference_forward, a mirror of the JS) must
    reproduce the Python ReservoirModel's own predictions exactly -- for both
    the digit-serial heads (+, -, *) and the single compare head (>)."""
    from mathfly.export_web import (train_for_viz, build_bundle, reference_forward,
                                    op_tokens)

    model, enc, conn = train_for_viz({"synthetic_n": 400, "n_readout": 200,
                                      "n_train": 300, "seed": 0})
    bundle = build_bundle(model, enc, conn)

    for op, a, b in [("+", 47, 38), ("-", 63, 29), ("*", 7, 8), (">", 5, 9),
                     ("+", 9, 6), ("-", 4, 4)]:
        ref_ans, *_ = reference_forward(bundle, op, a, b)
        aa, bb = (b, a) if (op == "-" and b > a) else (a, b)
        toks = op_tokens(op, aa, bb)
        if op == ">":
            py_ans = model.predict(toks, name=op)
        else:
            py_ans = model.predict_number(toks, name=op)
        assert ref_ans == py_ans, (op, a, b, ref_ans, py_ans)


def test_synapse_export_matches_net():
    """The exported synapse-trained bundle (with learned gains + per-neuron bias
    + fixed-width tokens) must reproduce the torch model's predictions."""
    import pytest
    torch = pytest.importorskip("torch")
    from mathfly.connectome import make_synthetic_connectome
    from mathfly.io_encoding import IOEncoder
    from mathfly import synapse_train as st
    from mathfly.export_web import OPS, op_answer, reference_forward

    conn = make_synthetic_connectome(n=300, seed=0).rescale_spectral_radius(1.1)
    max_answer = max(op_answer(s["op"], s["hi"], s["hi"]) for s in OPS if s["op"] != ">")
    enc = IOEncoder(conn.n, n_input=48, n_readout=120, n_answers=max_answer + 1,
                    n_digits=3, steps_per_token=4, seed=0)
    net = st._build_net(conn, enc, alpha=0.3)
    st._warmstart(net, conn, enc, 200, np.random.default_rng(0))
    acc = st._eval(net, enc, "cpu", n=40)
    bundle = st.bundle_from_torch(net, enc, conn, acc)

    net.eval()
    for op, a, b in [("+", 8, 7), ("-", 9, 3), ("*", 6, 4), (">", 5, 9), ("*", 7, 7)]:
        ref, *_ = reference_forward(bundle, op, a, b)
        aa, bb = (b, a) if (op == "-" and b > a) else (a, b)
        U = torch.tensor(enc.encode(st.fx_tokens(op, aa, bb))[None])
        out = net(U, op)
        if op == ">":
            pred = int(out.argmax(1))
        else:
            dig = out.argmax(2)[0].numpy()
            pred = int((dig * (10 ** np.arange(enc.n_digits))).sum())
        assert ref == pred, (op, a, b, ref, pred)


def test_calc_fly_learns_table():
    """The sustained one-hot calculator fly should master the single-digit
    tables on a small connectome, and its bundle should round-trip."""
    import pytest
    pytest.importorskip("torch")
    from mathfly.connectome import make_synthetic_connectome
    from mathfly.calc_fly import CalcFly, build_bundle
    from mathfly.render_gif import render_calc_gif  # import path sanity

    conn = make_synthetic_connectome(n=900, seed=0).rescale_spectral_radius(1.15)
    fly = CalcFly(conn, n_read=400, seed=0)
    accs = fly.train(hidden=128, epochs=600, verbose=False)
    # exhaustive over all 100 single-digit problems; should be near-perfect
    assert accs["+"] > 0.9 and accs["*"] > 0.9 and accs[">"] > 0.9, accs
    # Python decode matches the trained head
    assert fly.predict("+", 7, 8) == 15
    assert fly.predict("*", 6, 7) == 42
    bundle = build_bundle(fly)
    assert bundle["meta"]["N"] == conn.n and "movement" in bundle
    for op in ["+", "-", "*", ">"]:
        assert "mlp" in bundle["tasks"][op]


def test_sci_encode_decode_roundtrip():
    """Fixed-point / signed output encoding must round-trip for every op."""
    import math
    from mathfly.sci_fly import OPS, encode_out, decode_out
    cases = {"add": (47, 38), "sub": (12, 40), "mul": (7, 8), "div": (81, 9),
             "square": (12,), "sqrt": (81,), "sin": (30,), "cos": (60,),
             "log": (10,), "inv": (4,)}
    for op in OPS:
        val = op["fn"](*cases[op["name"]])
        sign, digits = encode_out(op, val)
        got = decode_out(op, sign, digits)
        assert abs(got - round(val, op["dec"])) < 10 ** (-op["dec"]) / 2 + 1e-9, (op["name"], val, got)


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
