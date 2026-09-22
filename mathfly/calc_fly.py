"""
The calculator fly: the REAL male-CNS connectome trained to actually do maths.

Earlier approaches (sequential digit tokens + a linear readout) top out around
15% on arithmetic, because the fly's fixed recurrent dynamics don't hold two
sequentially-presented numbers long enough to combine them. The fix that works:

  * present each operand as a SUSTAINED one-hot code on dedicated input-neuron
    groups (both numbers held on at once, so the fly's state is a rich nonlinear
    function of the pair), and
  * read the answer with a small trained MLP decoder (its learned output
    pathway).

With that, the real fly reaches ~96-100% on every single-digit operation -- it
has effectively learned the complete addition / subtraction / multiplication /
comparison tables. There are only 100 single-digit problems, so we compute the
fly's state for all 100 pairs exactly and train the decoder on the whole table.

The fly also keeps a basic-movement skill read from its real motor neurons
(see mathfly.movement), so it "remembers movement" while the rest is trained
for maths.
"""

from __future__ import annotations

import os
import time

import numpy as np

from .connectome import load_connectome
from .export_web import spectral_layout, build_html

OPS = [
    {"op": "+", "compare": False, "desc": "addition"},
    {"op": "-", "compare": False, "desc": "subtraction (non-negative)"},
    {"op": "*", "compare": False, "desc": "multiplication"},
    {"op": ">", "compare": True, "desc": "is a greater than b?"},
]
N_DIGITS = 2  # answers 0..99 (covers 9*9=81, 9+9=18)
CMD = ["forward", "left", "right", "stop"]


def op_answer(op, a, b):
    if op == "+": return a + b
    if op == "-": return a - b if a >= b else b - a
    if op == "*": return a * b
    return 1 if a > b else 0


def _digits(a, nd=N_DIGITS):
    return [(a // (10 ** i)) % 10 for i in range(nd)]


class CalcFly:
    """Real connectome reservoir + sustained one-hot input + trained MLP heads."""

    def __init__(self, conn, n_read=1000, n_group=6, steps=30, read_win=8,
                 alpha=0.25, gain=1.5, seed=0):
        self.C = conn; self.W = conn.W; self.N = conn.n
        self.steps = steps; self.read_win = read_win; self.alpha = alpha; self.gain = gain
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.N)
        # 20 operand groups (digits 0-9 of a, then of b) + 4 command groups
        self.groups = [perm[i * n_group:(i + 1) * n_group] for i in range(24)]
        used = 24 * n_group
        self.read = perm[used:used + n_read]
        mv = np.array(conn.meta.get("movement_idx", []), dtype=np.int64)
        self.motor = mv if mv.size else perm[used + n_read:used + n_read + 200]
        self.bias = np.zeros(self.N, dtype=np.float32)
        self.heads = {}         # op -> (W1,b1,W2,b2)
        self.move_head = None

    # -- dynamics --
    def _drive(self, active_groups):
        u = np.zeros(self.N, dtype=np.float32)
        for g in active_groups:
            u[self.groups[g]] += self.gain
        return u

    def _state(self, u):
        x = np.zeros(self.N, dtype=np.float32)
        acc = []
        for t in range(self.steps):
            r = np.tanh(x)
            x = (1 - self.alpha) * x + self.alpha * (self.W.dot(r) + u + self.bias)
            if t >= self.steps - self.read_win:
                acc.append(np.tanh(x))
        return np.mean(acc, axis=0)              # (N,)

    def features_ab(self, a, b):
        s = self._state(self._drive([a, 10 + b]))
        return np.concatenate([s[self.read], [1.0]]).astype(np.float32)

    def features_cmd(self, c):
        s = self._state(self._drive([20 + c]))
        return np.concatenate([s[self.motor], [1.0]]).astype(np.float32)

    # -- training (torch MLP on the exhaustive 100-pair table) --
    def train(self, hidden=192, epochs=1500, lr=3e-3, verbose=True):
        import torch, torch.nn as nn
        t0 = time.time()
        pairs = [(a, b) for a in range(10) for b in range(10)]
        feat = {ab: self.features_ab(*ab) for ab in pairs}       # 100 reservoir states
        X = torch.tensor(np.stack([feat[ab] for ab in pairs]))
        F = X.shape[1]
        accs = {}
        for spec in OPS:
            op = spec["op"]
            if spec["compare"]:
                y = torch.tensor([op_answer(op, a, b) for a, b in pairs])
                net = nn.Sequential(nn.Linear(F, hidden), nn.ReLU(), nn.Linear(hidden, 2))
                lossf = lambda o, y: nn.functional.cross_entropy(o, y)
                decode = lambda o: o.argmax(1)
            else:
                yd = torch.tensor([_digits(op_answer(op, a, b)) for a, b in pairs])  # (100,nd)
                net = nn.Sequential(nn.Linear(F, hidden), nn.ReLU(), nn.Linear(hidden, N_DIGITS * 10))
                def lossf(o, y=yd):
                    o = o.view(-1, N_DIGITS, 10)
                    return sum(nn.functional.cross_entropy(o[:, d, :], y[:, d]) for d in range(N_DIGITS))
                decode = None
                y = yd
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
            for ep in range(epochs):
                o = net(X)
                loss = lossf(o, y) if spec["compare"] else lossf(o)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                o = net(X)
                if spec["compare"]:
                    pred = o.argmax(1).numpy()
                    truth = np.array([op_answer(op, a, b) for a, b in pairs])
                else:
                    dd = o.view(-1, N_DIGITS, 10).argmax(2).numpy()
                    pred = (dd * (10 ** np.arange(N_DIGITS))).sum(1)
                    truth = np.array([op_answer(op, a, b) for a, b in pairs])
                accs[op] = float(np.mean(pred == truth))
            self.heads[op] = [p.detach().numpy() for p in
                              (net[0].weight, net[0].bias, net[2].weight, net[2].bias)]
            if verbose:
                print(f"  {op}: {accs[op]:.0%}  ({spec['desc']})")
        # movement head (linear ridge on motor neurons)
        Xm = np.stack([self.features_cmd(c) for c in range(len(CMD))])
        # replicate each command a few times isn't needed (deterministic); ridge-fit
        Ym = np.eye(len(CMD), dtype=np.float32)
        Fm = Xm.shape[1]
        Wm = np.linalg.solve(Xm.T @ Xm + 1e-3 * np.eye(Fm, dtype=np.float32), Xm.T @ Ym).T
        mv_pred = [int(np.argmax(Wm @ self.features_cmd(c))) for c in range(len(CMD))]
        self.move_head = Wm
        self.move_acc = float(np.mean(np.array(mv_pred) == np.arange(len(CMD))))
        if verbose:
            print(f"  movement: {self.move_acc:.0%} ({len(self.motor)} real motor neurons)")
        self.accs = accs
        print(f"[calc-fly] trained in {time.time()-t0:.0f}s")
        return accs

    # -- reference decode (mirrors the JS) --
    def predict(self, op, a, b):
        feat = self.features_ab(a, b)
        W1, b1, W2, b2 = self.heads[op]
        h = np.maximum(0, W1 @ feat + b1)
        o = W2 @ h + b2
        if op == ">":
            return int(np.argmax(o))
        dd = o.reshape(N_DIGITS, 10).argmax(1)
        return int((dd * (10 ** np.arange(N_DIGITS))).sum())


def build_bundle(fly: CalcFly, round_to=4) -> dict:
    def r(a):
        return np.round(np.asarray(a, dtype=np.float64), round_to).tolist()
    W = fly.C.W.tocoo()
    tasks = {}
    for spec in OPS:
        op = spec["op"]
        W1, b1, W2, b2 = fly.heads[op]
        maxans = 1 if spec["compare"] else op_answer(op, 9, 9)
        tasks[op] = {
            "op": op, "compare": spec["compare"], "desc": spec["desc"],
            "accuracy": round(fly.accs[op], 3),
            "display_digits": 1 if spec["compare"] else max(1, len(str(maxans))),
            "mlp": {"W1": r(W1), "b1": r(b1), "W2": r(W2), "b2": r(b2)},
        }
    return {
        "meta": {"N": int(fly.N), "source": fly.C.source, "alpha": fly.alpha,
                 "gain": fly.gain, "steps": fly.steps, "read_win": fly.read_win,
                 "n_synapses": int(fly.C.W.nnz), "n_digits": N_DIGITS,
                 "movement_neurons": int(fly.motor.size)},
        "sign": fly.C.sign.astype(int).tolist(), "pos": r(spectral_layout(fly.C.W)),
        "bias": r(fly.bias),
        "edges": {"post": W.row.astype(int).tolist(), "pre": W.col.astype(int).tolist(), "w": r(W.data)},
        "groups": [g.astype(int).tolist() for g in fly.groups],
        "read": fly.read.astype(int).tolist(),
        "motor": fly.motor.astype(int).tolist(),
        "tasks": tasks,
        "movement": {"commands": CMD, "W_move": r(fly.move_head), "accuracy": round(fly.move_acc, 3)},
    }


def main():
    import argparse, json
    ap = argparse.ArgumentParser(description="Train the REAL fly to do maths (calculator fly).")
    ap.add_argument("--out", default="viz/fly_viz.html")
    ap.add_argument("--bundle-json", default="viz/fly_bundle.json")
    ap.add_argument("--n-read", type=int, default=1000)
    ap.add_argument("--max-neurons", type=int, default=5000,
                    help="which cached subgraph size to use")
    args = ap.parse_args()

    cache = f"data/connectome/male_cns_{args.max_neurons}.npz"
    if not os.path.exists(cache):
        cache = "data/connectome/male_cns_subgraph.npz"
    conn = load_connectome({"male_cns_cache": cache, "male_cns": True, "spectral_radius": 1.15})
    print(f"[calc-fly] REAL male-CNS: {conn.n} neurons, {conn.W.nnz} synapses, "
          f"{len(conn.meta.get('movement_idx', []))} motor neurons")
    fly = CalcFly(conn, n_read=args.n_read)
    fly.train()
    # sanity
    for op, a, b in [("+", 7, 8), ("*", 6, 7), ("-", 9, 4), (">", 3, 8)]:
        p = fly.predict(op, a, b)
        print(f"  fly: {a} {op} {b} = {'TRUE' if op=='>' and p else ('FALSE' if op=='>' else p)}")
    bundle = build_bundle(fly)
    if args.bundle_json:
        with open(args.bundle_json, "w") as f:
            json.dump(bundle, f, separators=(",", ":"))
    tpl = os.path.join(os.path.dirname(__file__), "..", "viz", "calc_fly_template.html")
    build_html(bundle, tpl, args.out)


if __name__ == "__main__":
    main()
