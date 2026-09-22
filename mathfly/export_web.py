"""
Export a trained reservoir model to a self-contained interactive web page where
you can type a math problem and WATCH the fly's neurons fire as it solves it.

The page is not a canned animation: it embeds the real connectome, the frozen
recurrent weights, the sensory embedding, and the trained readout heads, and it
re-runs the *exact* reservoir dynamics in JavaScript, step by step. The answer
you see emerge is the model's genuine output.

Readout heads exported per operator:
  * "+", "-", "*"  ->  DIGIT-SERIAL heads (one 10-way classifier per decimal
                       place), so multi-digit answers work and the page can
                       accept multi-digit operands.
  * ">"            ->  a single binary head (is a > b?).

Pipeline:
    train a ReservoirModel on each operator's range  ->  bundle everything to
    JSON  ->  inline the JSON into an HTML template  ->  standalone fly_viz.html

CLI:
    python -m mathfly.export_web --out viz/fly_viz.html
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import sparse

from .connectome import load_connectome
from .io_encoding import IOEncoder, VOCAB, TOKEN2ID, tokenize_problem
from .model import ReservoirModel

# Operators shown in the visualiser, each with its trained operand range and
# readout mode. Ranges are chosen so the reservoir stays meaningful to watch.
# Operand ranges are chosen per operator to sit where a random-reservoir linear
# readout is genuinely competent: comparison of two-digit numbers is easy and
# accurate, so it stays 0..99; sequential digit-binding for multi-digit
# arithmetic is hard, so +,-,* use smaller operands but still produce genuine
# multi-digit ANSWERS decoded on the digit-serial odometer. The page shows each
# head's real held-out accuracy, so nothing is oversold.
OPS = [
    {"op": "+", "lo": 0, "hi": 12, "mode": "digits", "desc": "addition"},
    {"op": "-", "lo": 0, "hi": 15, "mode": "digits", "desc": "subtraction (non-negative)"},
    {"op": "*", "lo": 0, "hi": 9,  "mode": "digits", "desc": "multiplication"},
    {"op": ">", "lo": 0, "hi": 99, "mode": "single", "desc": "is a greater than b?"},
]
N_DIGITS = 4  # covers answers up to 9999 with 4x10 outputs


# --------------------------------------------------------------------------- #
# problem sampling / ground truth
# --------------------------------------------------------------------------- #

def op_answer(op, a, b):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    return 1 if a > b else 0            # ">"


def op_tokens(op, a, b):
    # compare is presented to the network as the "a - b =" stream.
    tok_op = "-" if op == ">" else op
    return tokenize_problem(a, tok_op, b)


def sample_problem(op, lo, hi, rng):
    a = int(rng.integers(lo, hi + 1)); b = int(rng.integers(lo, hi + 1))
    if op == "-" and b > a:
        a, b = b, a                     # keep subtraction non-negative
    return op_tokens(op, a, b), op_answer(op, a, b), a, b


# --------------------------------------------------------------------------- #
# layout + bundle
# --------------------------------------------------------------------------- #

def spectral_layout(W: sparse.csr_matrix, seed: int = 0) -> np.ndarray:
    """2D neuron layout: spectral embedding of the (symmetrised) connectivity,
    so wired-together neurons sit near each other. A per-axis RANK (quantile)
    transform spreads the points evenly across the field -- spectral coordinates
    are heavily skewed (a few outliers), so a raw min/max normalise collapses
    everything into a corner. Rank keeps the ordering/cluster structure while
    filling the frame. Falls back to random."""
    n = W.shape[0]
    A = abs(W); A = (A + A.T) * 0.5
    try:
        from scipy.sparse.linalg import eigsh
        d = np.asarray(A.sum(axis=1)).ravel()
        L = sparse.diags(d) - A
        vals, vecs = eigsh(L.astype(np.float64), k=6, sigma=0, which="LM")
        xy = vecs[:, np.argsort(vals)[1:3]].astype(np.float64)
    except Exception:
        xy = np.random.default_rng(seed).standard_normal((n, 2))
    # rank-normalise each axis to [0.03, 0.97] for an even, framed spread
    out = np.empty_like(xy)
    for d2 in range(2):
        order = np.argsort(xy[:, d2], kind="stable")
        ranks = np.empty(n); ranks[order] = np.arange(n)
        out[:, d2] = 0.03 + 0.94 * (ranks / max(1, n - 1))
    # a touch of deterministic jitter breaks the grid look
    rng = np.random.default_rng(seed)
    out += rng.uniform(-0.006, 0.006, size=out.shape)
    return np.clip(out, 0.01, 0.99).astype(np.float32)


def train_heads(model: ReservoirModel, enc: IOEncoder, n_train=3000, ridge=1e-2,
                seed=0, verbose=True):
    """Train a readout head per operator on its operand range, and measure
    held-out accuracy (stored on model.viz_acc for the page to display)."""
    rng = np.random.default_rng(seed)
    model.viz_acc = {}
    for spec in OPS:
        op = spec["op"]
        probs, answers = [], []
        for _ in range(n_train):
            tok, ans, _, _ = sample_problem(op, spec["lo"], spec["hi"], rng)
            probs.append(tok); answers.append(ans)
        X = np.stack([model.features(p) for p in probs]).astype(np.float32)
        answers = np.array(answers)
        if spec["mode"] == "digits":
            model.fit_digit_readout(X, answers, ridge=ridge, name=op)
        else:
            Y = np.stack([enc.target_onehot(a) for a in answers])
            model.fit_readout(X, Y, ridge=ridge, name=op)
        # held-out accuracy
        te_rng = np.random.default_rng(seed + 4242)
        tp, ta = [], []
        for _ in range(500):
            tok, ans, _, _ = sample_problem(op, spec["lo"], spec["hi"], te_rng)
            tp.append(tok); ta.append(ans)
        if spec["mode"] == "digits":
            pred = [model.predict_number(p, name=op) for p in tp]
        else:
            pred = [model.predict(p, name=op) for p in tp]
        acc = float(np.mean(np.array(pred) == np.array(ta)))
        model.viz_acc[op] = acc
        if verbose:
            print(f"  trained '{op}' head ({spec['mode']}, range {spec['lo']}..{spec['hi']})"
                  f"  held-out accuracy {acc:.0%}")


def build_bundle(model: ReservoirModel, enc: IOEncoder, conn, round_to=4) -> dict:
    W = conn.W.tocoo()

    def r(a):
        return np.round(np.asarray(a, dtype=np.float64), round_to).tolist()

    tasks = {}
    for spec in OPS:
        op = spec["op"]
        max_ans = op_answer(op, spec["hi"], spec["hi"]) if op != "-" else spec["hi"]
        if op == ">":
            max_ans = 1
        entry = {
            "op": op, "operand_lo": spec["lo"], "operand_hi": spec["hi"],
            "mode": spec["mode"], "compare": op == ">",
            "max_answer": int(max_ans),
            "display_digits": max(1, len(str(int(max_ans)))),
            "description": spec["desc"],
        }
        entry["accuracy"] = round(float(getattr(model, "viz_acc", {}).get(op, 0.0)), 3)
        if spec["mode"] == "digits":
            entry["digit_heads"] = r(model.digit_heads[op])   # (n_digits, 10, F)
            entry["n_digits"] = int(enc.n_digits)
        else:
            entry["W_out"] = r(model.readouts[op])            # (n_answers, F)
            entry["n_answers"] = int(model.readouts[op].shape[0])
        tasks[op] = entry

    pos = spectral_layout(conn.W)
    return {
        "meta": {
            "N": int(conn.n), "source": conn.source, "alpha": float(model.alpha),
            "steps_per_token": enc.steps_per_token, "settle_tokens": enc.settle_tokens,
            "input_gain": float(enc.input_gain), "n_input": int(enc.n_input),
            "n_readout": int(enc.n_readout), "n_synapses": int(conn.W.nnz),
            "n_digits": int(enc.n_digits),
        },
        "vocab": VOCAB, "token2id": TOKEN2ID,
        "input_neurons": enc.input_neurons.astype(int).tolist(),
        "readout_neurons": enc.readout_neurons.astype(int).tolist(),
        "embed": r(enc.embed),
        "sign": conn.sign.astype(int).tolist(),
        "pos": r(pos),
        "edges": {"post": W.row.astype(int).tolist(), "pre": W.col.astype(int).tolist(), "w": r(W.data)},
        "tasks": tasks,
    }


# --------------------------------------------------------------------------- #
# reference forward pass (mirror of the JS; used by tests + the GIF renderer)
# --------------------------------------------------------------------------- #

def reference_forward(bundle: dict, op: str, a: int, b: int):
    """Pure-NumPy re-implementation of the in-browser forward pass. Returns
    (answer, R, U, tokens, (win_start, T))."""
    m = bundle["meta"]
    N = m["N"]; alpha = m["alpha"]; spt = m["steps_per_token"]; settle = m["settle_tokens"]
    gain = m["input_gain"]
    embed = np.array(bundle["embed"], dtype=np.float32)
    inp = np.array(bundle["input_neurons"]); rd = np.array(bundle["readout_neurons"])
    t2i = bundle["token2id"]
    pre = np.array(bundle["edges"]["pre"]); post = np.array(bundle["edges"]["post"])
    w = np.array(bundle["edges"]["w"], dtype=np.float32)
    bias = np.array(bundle.get("bias", np.zeros(N)), dtype=np.float32)   # 0 for reservoir models
    tb = bundle["tasks"][op]

    if op == "-" and b > a:
        a, b = b, a
    pad = tb.get("pad_width")
    if pad:                                    # fixed-width tokenisation (synapse-trained models)
        tok_op = "-" if op == ">" else op
        tokens = list(str(int(a)).rjust(pad, "0")) + [tok_op] + list(str(int(b)).rjust(pad, "0")) + ["="]
    else:
        tokens = op_tokens(op, a, b)
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

    x = np.zeros(N, dtype=np.float32); R = np.empty((T, N), dtype=np.float32)
    for t in range(T):
        rr = np.tanh(x)
        rec = np.zeros(N, dtype=np.float32)
        np.add.at(rec, post, w * rr[pre])
        x = (1 - alpha) * x + alpha * (rec + U[t] + bias); R[t] = np.tanh(x)

    win_start = T - settle * spt
    feat = np.concatenate([R[win_start:][:, rd].mean(0), [1.0]]).astype(np.float32)
    if tb["mode"] == "digits":
        heads = np.array(tb["digit_heads"], dtype=np.float32)   # (D,10,F)
        digits = np.argmax(heads @ feat, axis=1)
        answer = int(sum(int(d) * (10 ** i) for i, d in enumerate(digits)))
    else:
        answer = int(np.argmax(np.array(tb["W_out"], dtype=np.float32) @ feat))
    return answer, R, U, tokens, (win_start, T)


# --------------------------------------------------------------------------- #
# html build
# --------------------------------------------------------------------------- #

def build_html(bundle: dict, template_path: str, out_path: str):
    with open(template_path) as f:
        html = f.read()
    payload = json.dumps(bundle, separators=(",", ":"))
    html = html.replace("/*__BUNDLE__*/null", payload)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[export] wrote {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB)")
    return out_path


def train_for_viz(config: dict):
    """Build connectome + encoder and train all viz heads. Returns model, enc, conn."""
    config = dict(config)
    config.setdefault("synthetic_n", 1400)
    config.setdefault("spectral_radius", 1.15)
    conn = load_connectome(config)
    max_answer = max(op_answer(s["op"], s["hi"], s["hi"]) for s in OPS if s["op"] != ">")
    enc = IOEncoder(conn.n, n_input=config.get("n_input", 96),
                    n_readout=config.get("n_readout", 512),
                    n_answers=max_answer + 1, n_digits=N_DIGITS,
                    steps_per_token=config.get("steps_per_token", 5),
                    input_gain=config.get("input_gain", 1.0), seed=config.get("seed", 0))
    model = ReservoirModel(conn, enc, alpha=config.get("alpha", 0.25), seed=config.get("seed", 0))
    print(f"[connectome] source={conn.source} N={conn.n} synapses={conn.W.nnz} "
          f"rho~={conn.spectral_radius():.2f}")
    train_heads(model, enc, n_train=config.get("n_train", 3000),
                ridge=config.get("ridge", 1e-2), seed=config.get("seed", 0))
    return model, enc, conn


def main():
    ap = argparse.ArgumentParser(description="Export an interactive fly-brain visualiser.")
    ap.add_argument("--config", help="JSON/YAML training config")
    ap.add_argument("--out", default="viz/fly_viz.html")
    ap.add_argument("--bundle-json", help="also write the raw bundle JSON here")
    ap.add_argument("--male-cns", action="store_true",
                    help="use the REAL male-CNS connectome subgraph (see scripts/download_male_cns.py)")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(__file__),
                    "..", "viz", "fly_viz_template.html"))
    args = ap.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f) if args.config.endswith(".json") else __import__("yaml").safe_load(f)
    if args.male_cns:
        config["male_cns"] = True
        config.setdefault("n_readout", 800)
        config.setdefault("spectral_radius", 1.15)

    model, enc, conn = train_for_viz(config)
    bundle = build_bundle(model, enc, conn)

    for op, a, b in [("+", 47, 38), ("-", 63, 29), ("*", 7, 8), (">", 5, 9)]:
        ans, *_ = reference_forward(bundle, op, a, b)
        shown = "TRUE" if (op == ">" and ans == 1) else ("FALSE" if op == ">" else ans)
        print(f"  fly says: {a} {op} {b} = {shown}")

    if args.bundle_json:
        with open(args.bundle_json, "w") as f:
            json.dump(bundle, f, separators=(",", ":"))
        print(f"[export] wrote bundle -> {args.bundle_json}")
    build_html(bundle, args.template, args.out)


if __name__ == "__main__":
    main()
