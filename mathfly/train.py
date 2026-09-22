"""
Training: teach the connectome network to do math, level by level.

Two entry points:

  train_reservoir(...)  -- NumPy path. For each curriculum level, run every
        problem through the fixed connectome reservoir, collect motor-neuron
        features, and fit the linear readout (ridge regression). Optionally
        carry a growing readout across levels (continual curriculum).

  train_bptt(...)       -- PyTorch path. Backprop-through-time trains the edge
        gains + readout with cross-entropy. Slower, stronger.

Run `python -m mathfly.train --help` for the CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .connectome import load_connectome
from .envs import CURRICULUM, CURRICULUM_BY_NAME, make_dataset
from .io_encoding import IOEncoder
from .model import ReservoirModel


def _accuracy(preds, answers):
    return float(np.mean(np.array(preds) == np.array(answers)))


def _collect_features(model: ReservoirModel, problems):
    return np.stack([model.features(toks) for toks in problems]).astype(np.float32)


def train_reservoir(config: dict):
    """Curriculum training of the readout on a fixed connectome reservoir."""
    t0 = time.time()
    conn = load_connectome(config)
    print(f"[connectome] source={conn.source} N={conn.n} synapses={conn.W.nnz} "
          f"rho~={conn.spectral_radius():.2f}")

    max_answer = max(t.max_answer for t in _selected_tasks(config)) + 1
    enc = IOEncoder(
        n_neurons=conn.n,
        n_input=config.get("n_input", 64),
        n_readout=config.get("n_readout", 256),
        n_answers=max_answer,
        steps_per_token=config.get("steps_per_token", 4),
        input_gain=config.get("input_gain", 1.0),
        seed=config.get("seed", 0),
    )
    model = ReservoirModel(conn, enc, alpha=config.get("alpha", 0.3),
                           noise=config.get("noise", 0.0), seed=config.get("seed", 0))

    n_train = config.get("n_train", 400)
    n_eval = config.get("n_eval", 200)
    ridge = config.get("ridge", 1e-2)
    results = []

    for task in _selected_tasks(config):
        tr_prob, tr_ans = make_dataset(task, n_train, seed=config.get("seed", 0))
        te_prob, te_ans = make_dataset(task, n_eval, seed=config.get("seed", 0) + 999)

        Xtr = _collect_features(model, tr_prob)
        Ytr = np.stack([enc.target_onehot(a) for a in tr_ans])
        model.fit_readout(Xtr, Ytr, ridge=ridge, name=task.name)

        tr_pred = [int(np.argmax(model.W_out @ x)) for x in Xtr]
        Xte = _collect_features(model, te_prob)
        te_pred = [int(np.argmax(model.W_out @ x)) for x in Xte]

        acc_tr, acc_te = _accuracy(tr_pred, tr_ans), _accuracy(te_pred, te_ans)
        # chance baseline for context
        chance = 1.0 / (task.max_answer + 1)
        print(f"[{task.name:12s}] train={acc_tr:5.1%} test={acc_te:5.1%} "
              f"(chance {chance:4.1%})  :: {task.description}")
        results.append({"task": task.name, "train_acc": acc_tr, "test_acc": acc_te,
                        "chance": chance})

    out_dir = config.get("out_dir", "runs/reservoir")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "readout.npz"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"config": config, "results": results,
                   "seconds": round(time.time() - t0, 1)}, f, indent=2)
    print(f"[done] {time.time()-t0:.1f}s  ->  {out_dir}")
    return model, results


def train_reservoir_digits(config: dict):
    """Curriculum training with DIGIT-SERIAL readout: each answer is decoded
    digit-by-digit, so large answers are no longer capped by a classifier size.
    Metrics report both exact-match and per-digit accuracy."""
    t0 = time.time()
    conn = load_connectome(config)
    print(f"[connectome] source={conn.source} N={conn.n} synapses={conn.W.nnz} "
          f"rho~={conn.spectral_radius():.2f}")

    tasks = _selected_tasks(config)
    max_ans = max(t.max_answer for t in tasks)
    n_digits = config.get("n_digits", max(1, len(str(max_ans))))
    enc = IOEncoder(
        n_neurons=conn.n, n_input=config.get("n_input", 64),
        n_readout=config.get("n_readout", 256), n_answers=1,
        n_digits=n_digits, steps_per_token=config.get("steps_per_token", 4),
        input_gain=config.get("input_gain", 1.0), seed=config.get("seed", 0),
    )
    model = ReservoirModel(conn, enc, alpha=config.get("alpha", 0.3),
                           noise=config.get("noise", 0.0), seed=config.get("seed", 0))

    n_train = config.get("n_train", 400)
    n_eval = config.get("n_eval", 200)
    ridge = config.get("ridge", 1e-2)
    results = []
    print(f"[digit-serial] {n_digits} digit heads (answers up to {10**n_digits - 1})")
    for task in tasks:
        tr_prob, tr_ans = make_dataset(task, n_train, seed=config.get("seed", 0))
        te_prob, te_ans = make_dataset(task, n_eval, seed=config.get("seed", 0) + 999)
        Xtr = _collect_features(model, tr_prob)
        model.fit_digit_readout(Xtr, tr_ans, ridge=ridge, name=task.name)

        te_pred = [model.predict_number(p, name=task.name) for p in te_prob]
        exact = _accuracy(te_pred, te_ans)
        # per-digit accuracy
        pd = np.array([enc.target_digits(p) for p in te_pred])
        td = np.array([enc.target_digits(a) for a in te_ans])
        digit_acc = float(np.mean(pd == td))
        print(f"[{task.name:12s}] exact={exact:5.1%} per_digit={digit_acc:5.1%}"
              f"  :: {task.description}")
        results.append({"task": task.name, "exact_acc": exact, "digit_acc": digit_acc})

    out_dir = config.get("out_dir", "runs/digits")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "readout.npz"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"config": config, "results": results, "mode": "digits",
                   "seconds": round(time.time() - t0, 1)}, f, indent=2)
    print(f"[done] {time.time()-t0:.1f}s  ->  {out_dir}")
    return model, results


def train_bptt(config: dict):
    """Backprop-through-time on a single task (or the last curriculum level)."""
    import torch
    import torch.nn.functional as F

    conn = load_connectome(config)
    task = CURRICULUM_BY_NAME[config.get("task", "add_1digit")]
    enc = IOEncoder(conn.n, n_input=config.get("n_input", 64),
                    n_readout=config.get("n_readout", 256),
                    n_answers=task.max_answer + 1,
                    steps_per_token=config.get("steps_per_token", 4),
                    seed=config.get("seed", 0))
    from .model import build_torch_rnn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_torch_rnn(conn, enc, alpha=config.get("alpha", 0.3)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=config.get("lr", 3e-3))

    n_iters = config.get("n_iters", 300)
    batch = config.get("batch", 32)
    rng = np.random.default_rng(config.get("seed", 0))
    for it in range(n_iters):
        probs, ans = [], []
        for _ in range(batch):
            toks, a = task.sample(rng)
            probs.append(enc.encode(toks)); ans.append(a)
        T = max(u.shape[0] for u in probs)
        U = np.zeros((batch, T, conn.n), dtype=np.float32)
        for i, u in enumerate(probs):
            U[i, -u.shape[0]:] = u                       # right-align
        U = torch.tensor(U, device=device)
        y = torch.tensor(ans, device=device)
        logits = net(U)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 25 == 0 or it == n_iters - 1:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"[bptt {it:4d}] loss={loss.item():.3f} batch_acc={acc:.1%}")
    out_dir = config.get("out_dir", "runs/bptt")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(out_dir, "net.pt"))
    print(f"[done] saved -> {out_dir}/net.pt")
    return net


def _selected_tasks(config):
    names = config.get("tasks")
    if not names:
        return CURRICULUM
    return [CURRICULUM_BY_NAME[n] for n in names]


def _load_config(path):
    if not path:
        return {}
    with open(path) as f:
        if path.endswith(".json"):
            return json.load(f)
        import yaml
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser(description="Train the Math-Fly connectome network.")
    p.add_argument("--config", help="JSON/YAML config file")
    p.add_argument("--mode", choices=["reservoir", "digits", "bptt", "rl"],
                   default="reservoir")
    p.add_argument("--tasks", nargs="*", help="subset of curriculum task names")
    p.add_argument("--synthetic-n", type=int, help="synthetic connectome size")
    p.add_argument("--n-train", type=int)
    p.add_argument("--n-eval", type=int)
    p.add_argument("--task", help="single task for bptt mode")
    p.add_argument("--out-dir")
    args = p.parse_args()

    config = _load_config(args.config)
    # command-line overrides of config-file values
    if args.tasks: config["tasks"] = args.tasks
    if args.synthetic_n: config["synthetic_n"] = args.synthetic_n
    if args.n_train: config["n_train"] = args.n_train
    if args.n_eval: config["n_eval"] = args.n_eval
    if args.task: config["task"] = args.task
    if args.out_dir: config["out_dir"] = args.out_dir

    if args.mode == "reservoir":
        train_reservoir(config)
    elif args.mode == "digits":
        train_reservoir_digits(config)
    elif args.mode == "rl":
        from .rl_train import train_rl
        train_rl(config)
    else:
        train_bptt(config)


if __name__ == "__main__":
    main()
