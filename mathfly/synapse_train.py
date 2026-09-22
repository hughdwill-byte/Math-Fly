"""
Synapse training (backprop-through-time) on the REAL male-CNS connectome.

Where the reservoir path freezes the fly's wiring and trains only a readout,
this path also *tunes the synapses*: every real connection keeps its sign
(Dale's law) and the wiring stays exactly as the fly has it, but a positive gain
on each synapse, a per-neuron bias, and the readout heads are all learned end to
end by gradient descent through the recurrent dynamics. That is what lets the
network get meaningfully smarter than the fixed-reservoir version.

The trained network exports to the same interactive visualiser as everything
else, so you can watch the *synapse-trained real fly* solve problems.

Everything here is plain PyTorch ("torch"); it is installed automatically and
you don't need to know anything about it -- just run:

    python -m mathfly.synapse_train --out viz/fly_viz.html
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .connectome import load_connectome
from .io_encoding import IOEncoder, VOCAB, TOKEN2ID
from .export_web import OPS, N_DIGITS, sample_problem, op_answer, build_html
from .connectome import Connectome


def _build_net(conn: Connectome, enc: IOEncoder, alpha: float):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FlyNet(nn.Module):
        """Real connectome + learnable synapse gains, bias, and per-op heads."""

        def __init__(self):
            super().__init__()
            self.enc = enc; self.alpha = alpha; self.N = conn.n
            coo = conn.W.tocoo()
            self.register_buffer("edge_index", torch.tensor(
                np.vstack([coo.row, coo.col]), dtype=torch.long))
            self.register_buffer("edge_sign", torch.tensor(np.sign(coo.data), dtype=torch.float32))
            init = np.log(np.expm1(np.abs(coo.data) + 1e-3)).astype(np.float32)
            self.edge_loggain = nn.Parameter(torch.tensor(init))     # softplus -> >=0 gain
            self.bias = nn.Parameter(torch.zeros(self.N))
            self.register_buffer("out_neurons", torch.tensor(enc.readout_neurons, dtype=torch.long))
            # one head per operator; digit ops predict n_digits x 10, compare 2.
            self.heads = nn.ModuleDict()
            for spec in OPS:
                if spec["mode"] == "digits":
                    self.heads[spec["op"]] = nn.Linear(enc.n_readout, enc.n_digits * 10)
                else:
                    self.heads[spec["op"]] = nn.Linear(enc.n_readout, 2)

        def W(self):
            import torch
            vals = self.edge_sign * F.softplus(self.edge_loggain)
            return torch.sparse_coo_tensor(self.edge_index, vals, (self.N, self.N)).coalesce()

        def features(self, U):
            import torch
            W = self.W(); B, T, N = U.shape
            x = torch.zeros(B, N, device=U.device); a = self.alpha
            win = self.enc.readout_window(T); acc = []
            for t in range(T):
                r = torch.tanh(x)
                rec = torch.sparse.mm(W, r.t()).t()
                x = (1 - a) * x + a * (rec + U[:, t, :] + self.bias)
                if t >= win.start:
                    acc.append(torch.tanh(x)[:, self.out_neurons])
            return torch.stack(acc, 0).mean(0)                       # (B, n_readout)

        def forward(self, U, op):
            feat = self.features(U)
            out = self.heads[op](feat)
            if op != ">":
                out = out.view(out.shape[0], self.enc.n_digits, 10)
            return out

    return FlyNet()


# Fixed-width tokenisation: pad every operand to PAD_WIDTH digits so all problems
# tokenise to the same length. This keeps batched training identical to
# single-problem inference (no variable-length padding to reconcile) and gives
# the network a stable positional layout. Stored as pad_width in the bundle so
# the visualiser tokenises the same way.
PAD_WIDTH = 2


def fx_tokens(op, a, b, width=PAD_WIDTH):
    sa = str(int(a)).rjust(width, "0"); sb = str(int(b)).rjust(width, "0")
    return list(sa) + [("-" if op == ">" else op)] + list(sb) + ["="]


def _encode_batch(enc, op, lo, hi, n, rng, N):
    import torch
    probs, ans = [], []
    for _ in range(n):
        _, a, aa, bb = sample_problem(op, lo, hi, rng)
        probs.append(enc.encode(fx_tokens(op, aa, bb))); ans.append(a)
    T = probs[0].shape[0]                       # all equal length now
    U = np.zeros((n, T, N), dtype=np.float32)
    for i, u in enumerate(probs):
        U[i] = u
    return torch.tensor(U), np.array(ans)


def train(config: dict):
    import torch
    import torch.nn.functional as F

    t0 = time.time()
    conn = load_connectome({**config, "male_cns": config.get("male_cns", True)})
    print(f"[connectome] source={conn.source} N={conn.n} synapses={conn.W.nnz}")
    max_answer = max(op_answer(s["op"], s["hi"], s["hi"]) for s in OPS if s["op"] != ">")
    enc = IOEncoder(conn.n, n_input=config.get("n_input", 96),
                    n_readout=config.get("n_readout", 384), n_answers=max_answer + 1,
                    n_digits=config.get("n_digits", 3),
                    steps_per_token=config.get("steps_per_token", 5),
                    seed=config.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _build_net(conn, enc, alpha=config.get("alpha", 0.3)).to(device)
    rng = np.random.default_rng(config.get("seed", 0))

    # Warm-start the readout heads from the fast fixed-synapse reservoir solution.
    # At init the net's synapse gains equal the connectome weights and bias is 0,
    # so the pre-training accuracy below IS the fixed-wiring baseline -- the bar
    # that training the synapses then has to beat.
    _warmstart(net, conn, enc, config.get("warmstart_n", 1500), rng)
    base_acc = _eval(net, enc, device)
    print("[baseline: fixed synapses]  " + "  ".join(f"{op} {a:.0%}" for op, a in base_acc.items()))

    # Gentle LR on the synapses/bias, and keep gradients clipped: BPTT through a
    # 30-step recurrent net is easy to destabilise, and we're fine-tuning from a
    # good warm start, so small careful steps beat big ones.
    lr = config.get("lr", 8e-4)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    clip = config.get("clip", 1.0)
    rounds = config.get("rounds", 12)          # curriculum passes over all ops
    iters = config.get("iters_per_op", 20)     # updates per op per round
    batch = config.get("batch", 24)
    print(f"[synapse-train] {rounds} rounds x {len(OPS)} ops x {iters} iters, "
          f"batch {batch}, lr {lr}, device {device}")

    import copy
    best = {op: base_acc[op] for op in base_acc}
    best_state = copy.deepcopy(net.state_dict())   # warm start is the floor
    for rd in range(rounds):
        for spec in OPS:
            op = spec["op"]
            for _ in range(iters):
                U, ans = _encode_batch(enc, op, spec["lo"], spec["hi"], batch, rng, conn.n)
                U = U.to(device)
                out = net(U, op)
                if op == ">":
                    y = torch.tensor(ans, device=device)
                    loss = F.cross_entropy(out, y)
                else:
                    dig = np.stack([enc.target_digits(int(a)) for a in ans])  # (B, n_digits)
                    y = torch.tensor(dig, device=device)
                    loss = sum(F.cross_entropy(out[:, d, :], y[:, d]) for d in range(enc.n_digits))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
                opt.step()
        acc = _eval(net, enc, device)
        # keep the best-so-far weights (mean accuracy), so training never ships
        # something worse than the warm start
        if np.mean(list(acc.values())) >= np.mean(list(best.values())):
            best = dict(acc); best_state = copy.deepcopy(net.state_dict())
        print(f"[round {rd+1:2d}/{rounds}]  " +
              "  ".join(f"{op} {a:.0%}" for op, a in acc.items()) +
              f"   ({time.time()-t0:.0f}s)")

    # restore best-so-far weights before exporting
    if best_state is not None:
        net.load_state_dict(best_state)
    final = _eval(net, enc, device)
    print("[synapse training: fixed -> trained]")
    for op in base_acc:
        arrow = "up" if final[op] > base_acc[op] else ("=" if final[op] == base_acc[op] else "dn")
        print(f"   {op}:  {base_acc[op]:.0%}  ->  {final[op]:.0%}  [{arrow}]")
    bundle = bundle_from_torch(net, enc, conn, final)
    out_html = config.get("out", "viz/fly_viz.html")
    if config.get("bundle_json"):
        import json
        with open(config["bundle_json"], "w") as f:
            json.dump(bundle, f, separators=(",", ":"))
    import os
    tpl = os.path.join(os.path.dirname(__file__), "..", "viz", "fly_viz_template.html")
    build_html(bundle, tpl, out_html)
    print(f"[done] synapse-trained REAL fly in {time.time()-t0:.0f}s")
    return net, bundle


def _warmstart(net, conn, enc, n_train, rng):
    """Initialise each head from a closed-form ridge fit on the fixed-synapse
    reservoir features (using the same fixed-width tokens), so the network
    starts at reservoir-level accuracy and synapse training only improves it."""
    import torch
    from .model import ReservoirModel
    rm = ReservoirModel(conn, enc, alpha=net.alpha)
    for spec in OPS:
        op = spec["op"]
        P, A = [], []
        for _ in range(n_train):
            _, a, aa, bb = sample_problem(op, spec["lo"], spec["hi"], rng)
            P.append(fx_tokens(op, aa, bb)); A.append(a)
        X = np.stack([rm.features(p) for p in P]).astype(np.float32); A = np.array(A)
        if spec["mode"] == "digits":
            stack = rm.fit_digit_readout(X, A, name=op)          # (n_digits, 10, F)
            W = stack[:, :, :-1].reshape(enc.n_digits * 10, enc.n_readout)
            b = stack[:, :, -1].reshape(enc.n_digits * 10)
        else:
            # compare: direct 2-class ridge (not enc.n_answers-wide)
            Y = np.zeros((len(A), 2), dtype=np.float32); Y[np.arange(len(A)), A.astype(int)] = 1.0
            F = X.shape[1]
            WB = np.linalg.solve(X.T @ X + 1e-2 * np.eye(F, dtype=np.float32), X.T @ Y).T
            W = WB[:, :-1]; b = WB[:, -1]
        with torch.no_grad():
            net.heads[op].weight.copy_(torch.tensor(W, dtype=torch.float32))
            net.heads[op].bias.copy_(torch.tensor(b, dtype=torch.float32))


def _eval(net, enc, device, n=400):
    import torch
    net.eval()
    accs = {}
    rng = np.random.default_rng(4242)
    with torch.no_grad():
        for spec in OPS:
            op = spec["op"]
            U, ans = _encode_batch(enc, op, spec["lo"], spec["hi"], n, rng, net.N)
            out = net(U.to(device), op)
            if op == ">":
                pred = out.argmax(1).cpu().numpy()
            else:
                dig = out.argmax(2).cpu().numpy()                    # (n, n_digits)
                pred = (dig * (10 ** np.arange(enc.n_digits))).sum(1)
            accs[op] = float(np.mean(pred == ans))
    net.train()
    return accs


def bundle_from_torch(net, enc, conn, acc) -> dict:
    """Serialise the synapse-trained torch model into the visualiser bundle,
    including the learned synapse gains and per-neuron bias."""
    import torch
    import torch.nn.functional as F
    from .export_web import spectral_layout

    def r(a, nd=4):
        return np.round(np.asarray(a, dtype=np.float64), nd).tolist()

    with torch.no_grad():
        gains = (net.edge_sign * F.softplus(net.edge_loggain)).cpu().numpy()
        ei = net.edge_index.cpu().numpy()
        bias = net.bias.cpu().numpy()

    tasks = {}
    for spec in OPS:
        op = spec["op"]
        max_ans = 1 if op == ">" else (op_answer(op, spec["hi"], spec["hi"]) if op != "-" else spec["hi"])
        entry = {"op": op, "operand_lo": spec["lo"], "operand_hi": spec["hi"],
                 "mode": spec["mode"], "compare": op == ">", "max_answer": int(max_ans),
                 "display_digits": max(1, len(str(int(max_ans)))), "pad_width": PAD_WIDTH,
                 "description": spec["desc"], "accuracy": round(float(acc.get(op, 0)), 3)}
        lin = net.heads[op]
        Wt = lin.weight.detach().cpu().numpy()      # (out, n_readout)
        bt = lin.bias.detach().cpu().numpy()        # (out,)
        # fold Linear bias into an extra feature column: feat = [mean_rates, 1]
        WB = np.concatenate([Wt, bt[:, None]], axis=1)   # (out, n_readout+1)
        if spec["mode"] == "digits":
            entry["digit_heads"] = r(WB.reshape(enc.n_digits, 10, enc.n_readout + 1))
            entry["n_digits"] = int(enc.n_digits)
        else:
            entry["W_out"] = r(WB); entry["n_answers"] = int(WB.shape[0])
        tasks[op] = entry

    pos = spectral_layout(conn.W)
    return {
        "meta": {"N": int(conn.n), "source": conn.source + " (synapse-trained)",
                 "alpha": float(net.alpha), "steps_per_token": enc.steps_per_token,
                 "settle_tokens": enc.settle_tokens, "input_gain": float(enc.input_gain),
                 "n_input": int(enc.n_input), "n_readout": int(enc.n_readout),
                 "n_synapses": int(conn.W.nnz), "n_digits": int(enc.n_digits)},
        "vocab": VOCAB, "token2id": TOKEN2ID,
        "input_neurons": enc.input_neurons.astype(int).tolist(),
        "readout_neurons": enc.readout_neurons.astype(int).tolist(),
        "embed": r(enc.embed), "sign": conn.sign.astype(int).tolist(),
        "bias": r(bias), "pos": r(pos),
        "edges": {"post": ei[0].astype(int).tolist(), "pre": ei[1].astype(int).tolist(), "w": r(gains)},
        "tasks": tasks,
    }


def main():
    ap = argparse.ArgumentParser(description="Train the synapses of the real male-CNS fly.")
    ap.add_argument("--out", default="viz/fly_viz.html")
    ap.add_argument("--bundle-json", default=None)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--iters-per-op", type=int, default=20)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n-readout", type=int, default=800)
    ap.add_argument("--n-input", type=int, default=120)
    ap.add_argument("--synthetic", action="store_true", help="use synthetic connectome instead of real")
    args = ap.parse_args()
    config = {"out": args.out, "bundle_json": args.bundle_json, "rounds": args.rounds,
              "iters_per_op": args.iters_per_op, "batch": args.batch, "lr": args.lr,
              "n_readout": args.n_readout, "n_input": args.n_input,
              "spectral_radius": 1.15, "alpha": 0.25,
              "male_cns": not args.synthetic}
    train(config)


if __name__ == "__main__":
    main()
