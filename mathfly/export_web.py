"""
Export a trained reservoir model to a self-contained interactive web page where
you can type a math problem and WATCH the fly's neurons fire as it solves it.

The page is not a canned animation: it embeds the real connectome, the frozen
recurrent weights, the sensory embedding, and the trained readout heads, and it
re-runs the *exact* reservoir dynamics in JavaScript, step by step. The answer
you see emerge is the model's genuine output.

Pipeline:
    train (or load) a ReservoirModel  ->  bundle everything to JSON  ->
    inline the JSON into an HTML template  ->  standalone fly_viz.html

CLI:
    python -m mathfly.export_web --config configs/quickstart.yaml --out viz/fly_viz.html
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import sparse

from .connectome import load_connectome
from .envs import CURRICULUM_BY_NAME
from .io_encoding import IOEncoder, VOCAB, TOKEN2ID
from .model import ReservoirModel
from .train import train_reservoir

# Tasks exposed in the visualiser (single-digit operands, clean to demo).
VIZ_TASKS = ["compare", "add_1digit", "sub_1digit", "mul_1digit"]
OP_OF_TASK = {"compare": ">", "add_1digit": "+", "sub_1digit": "-", "mul_1digit": "*"}


def spectral_layout(W: sparse.csr_matrix, seed: int = 0) -> np.ndarray:
    """2D neuron layout: spectral embedding of the (symmetrised) connectivity,
    so wired-together neurons sit near each other and signal flow reads left to
    right. Falls back to a random layout if the eigensolve fails."""
    n = W.shape[0]
    A = abs(W)
    A = (A + A.T) * 0.5
    try:
        from scipy.sparse.linalg import eigsh
        d = np.asarray(A.sum(axis=1)).ravel()
        L = sparse.diags(d) - A
        # smallest eigenvectors of L (skip the trivial constant one)
        vals, vecs = eigsh(L.astype(np.float64), k=4, sigma=0, which="LM")
        order = np.argsort(vals)
        xy = vecs[:, order[1:3]]
    except Exception:
        rng = np.random.default_rng(seed)
        xy = rng.standard_normal((n, 2))
    # normalise to [0,1] with a margin
    xy = xy - xy.min(0)
    span = np.where(xy.max(0) > 0, xy.max(0), 1.0)
    xy = xy / span
    return (0.04 + 0.92 * xy).astype(np.float32)


def build_bundle(model: ReservoirModel, enc: IOEncoder, conn, tasks=VIZ_TASKS,
                 round_to: int = 4) -> dict:
    """Serialise everything the browser needs to reproduce the dynamics."""
    W = conn.W.tocoo()

    def r(a):  # compact float rounding
        return np.round(np.asarray(a, dtype=np.float64), round_to).tolist()

    task_bundles = {}
    for name in tasks:
        if name not in model.readouts:
            continue
        t = CURRICULUM_BY_NAME[name]
        lo, hi = 0, 9
        task_bundles[name] = {
            "op": OP_OF_TASK.get(name, "?"),
            "operand_lo": lo, "operand_hi": hi,
            "n_answers": int(model.readouts[name].shape[0]),
            "W_out": r(model.readouts[name]),          # (n_answers, n_readout+1)
            "compare": name == "compare",
            "description": t.description,
        }

    pos = spectral_layout(conn.W)
    bundle = {
        "meta": {
            "N": int(conn.n), "source": conn.source, "alpha": float(model.alpha),
            "steps_per_token": enc.steps_per_token, "settle_tokens": enc.settle_tokens,
            "input_gain": float(enc.input_gain), "n_input": int(enc.n_input),
            "n_readout": int(enc.n_readout), "n_synapses": int(conn.W.nnz),
        },
        "vocab": VOCAB, "token2id": TOKEN2ID,
        "input_neurons": enc.input_neurons.astype(int).tolist(),
        "readout_neurons": enc.readout_neurons.astype(int).tolist(),
        "embed": r(enc.embed),                          # (VOCAB, n_input)
        "sign": conn.sign.astype(int).tolist(),
        "pos": r(pos),                                  # (N, 2)
        "edges": {
            "post": W.row.astype(int).tolist(),
            "pre": W.col.astype(int).tolist(),
            "w": r(W.data),
        },
        "tasks": task_bundles,
    }
    return bundle


def reference_forward(bundle: dict, task: str, a: int, b: int):
    """Pure-NumPy re-implementation of the JS forward pass. Used by tests to
    guarantee the browser port matches the Python model. Returns (answer, R, U,
    tokens, readout_slice)."""
    m = bundle["meta"]
    N = m["N"]; alpha = m["alpha"]; spt = m["steps_per_token"]
    settle = m["settle_tokens"]; gain = m["input_gain"]
    embed = np.array(bundle["embed"], dtype=np.float32)
    inp = np.array(bundle["input_neurons"])
    rd = np.array(bundle["readout_neurons"])
    t2i = bundle["token2id"]
    pre = np.array(bundle["edges"]["pre"]); post = np.array(bundle["edges"]["post"])
    w = np.array(bundle["edges"]["w"], dtype=np.float32)

    tb = bundle["tasks"][task]
    # compare is trained on the "a - b =" token stream (answer = 1 if a>b else 0);
    # the ">" is only how we DISPLAY it.
    tok_op = "-" if tb["compare"] else tb["op"]
    tokens = list(str(a)) + [tok_op] + list(str(b)) + ["="]

    T = (len(tokens) + settle) * spt
    U = np.zeros((T, N), dtype=np.float32)
    ti = 0
    for tok in tokens:
        vec = embed[t2i[tok]] * gain
        for _ in range(spt):
            U[ti, inp] = vec; ti += 1
    go = embed[t2i["<GO>"]] * gain
    for _ in range(settle * spt):
        U[ti, inp] = go; ti += 1

    x = np.zeros(N, dtype=np.float32)
    R = np.empty((T, N), dtype=np.float32)
    for t in range(T):
        rr = np.tanh(x)
        rec = np.zeros(N, dtype=np.float32)
        np.add.at(rec, post, w * rr[pre])
        x = (1 - alpha) * x + alpha * (rec + U[t])
        R[t] = np.tanh(x)

    win_start = T - settle * spt
    feat = np.concatenate([R[win_start:][:, rd].mean(0), [1.0]]).astype(np.float32)
    W_out = np.array(tb["W_out"], dtype=np.float32)
    answer = int(np.argmax(W_out @ feat))
    return answer, R, U, tokens, (win_start, T)


def build_html(bundle: dict, template_path: str, out_path: str):
    with open(template_path) as f:
        html = f.read()
    payload = json.dumps(bundle, separators=(",", ":"))
    html = html.replace("/*__BUNDLE__*/null", payload)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    size = os.path.getsize(out_path) / 1e6
    print(f"[export] wrote {out_path} ({size:.2f} MB)")
    return out_path


def _load_or_train(config):
    """Return (model, enc, conn). Trains a fresh reservoir on the viz tasks."""
    config = dict(config)
    config.setdefault("synthetic_n", 700)
    config["tasks"] = [t for t in VIZ_TASKS]
    model, _ = train_reservoir(config)
    # rebuild the enc/conn the trainer used (deterministic from config)
    conn = load_connectome(config)
    max_answer = max(CURRICULUM_BY_NAME[t].max_answer for t in VIZ_TASKS) + 1
    enc = IOEncoder(conn.n, n_input=config.get("n_input", 64),
                    n_readout=config.get("n_readout", 256), n_answers=max_answer,
                    steps_per_token=config.get("steps_per_token", 4),
                    input_gain=config.get("input_gain", 1.0), seed=config.get("seed", 0))
    return model, enc, conn


def main():
    ap = argparse.ArgumentParser(description="Export an interactive fly-brain visualiser.")
    ap.add_argument("--config", help="JSON/YAML training config")
    ap.add_argument("--out", default="viz/fly_viz.html")
    ap.add_argument("--bundle-json", help="also write the raw bundle JSON here")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(__file__),
                    "..", "viz", "fly_viz_template.html"))
    args = ap.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f) if args.config.endswith(".json") else __import__("yaml").safe_load(f)

    model, enc, conn = _load_or_train(config)
    bundle = build_bundle(model, enc, conn)

    # sanity: show a few real predictions that the page will reproduce
    for a, b, name in [(3, 4, "add_1digit"), (9, 2, "sub_1digit"), (6, 7, "mul_1digit")]:
        ans, *_ = reference_forward(bundle, name, a, b)
        print(f"  fly says: {a} {bundle['tasks'][name]['op']} {b} = {ans}")

    if args.bundle_json:
        with open(args.bundle_json, "w") as f:
            json.dump(bundle, f, separators=(",", ":"))
        print(f"[export] wrote bundle -> {args.bundle_json}")
    build_html(bundle, args.template, args.out)


if __name__ == "__main__":
    main()
