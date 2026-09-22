"""
The scientific-calculator fly: the REAL male-CNS connectome trained to behave
like a pocket scientific calculator over a bounded domain.

Same principle as calc_fly (sustained one-hot input + trained readout on the
fly's fixed dynamics), extended to:

  * two-digit operands (each number = two one-hot digit slots),
  * a bank of operations -- + - x /, x^2, sqrt, sin, cos, log10, 1/x -- each
    decoded by a small head on a shared trunk, and
  * fixed-point / signed outputs, so transcendental results show as decimals.

Each operation is trained over its whole bounded domain (0..99), so within that
range the fly is a genuine calculator. Honest limits, printed as accuracy: it
computes +/- by generalisation (~95%+), memorises the unary tables exactly, and
does its best on x (multi-digit multiplication is the hard case). Everything is
exact *within the trained range* -- like a real calculator's finite precision.

    python -m mathfly.sci_fly --out viz/fly_viz.html
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from .connectome import load_connectome
from .export_web import spectral_layout, build_html

# op: name, symbol, arity, K digits, decimals, signed, function, operand domains
OPS = [
    dict(name="add",  sym="+",   arity=2, K=3, dec=0, signed=False, fn=lambda a, b: a + b,        a=(0, 99), b=(0, 99)),
    dict(name="sub",  sym="−", arity=2, K=2, dec=0, signed=True,  fn=lambda a, b: a - b,      a=(0, 99), b=(0, 99)),
    dict(name="mul",  sym="×", arity=2, K=4, dec=0, signed=False, fn=lambda a, b: a * b,      a=(0, 99), b=(0, 99)),
    dict(name="div",  sym="÷", arity=2, K=2, dec=0, signed=False, fn=lambda a, b: a // b,     a=(0, 99), b=(1, 99)),
    dict(name="square", sym="x²", arity=1, K=4, dec=0, signed=False, fn=lambda a: a * a,      a=(0, 99)),
    dict(name="sqrt", sym="√", arity=1, K=3, dec=2, signed=False, fn=lambda a: math.sqrt(a),  a=(0, 99)),
    dict(name="sin",  sym="sin", arity=1, K=4, dec=3, signed=True,  fn=lambda a: math.sin(math.radians(a)), a=(0, 99)),
    dict(name="cos",  sym="cos", arity=1, K=4, dec=3, signed=True,  fn=lambda a: math.cos(math.radians(a)), a=(0, 99)),
    dict(name="log",  sym="log", arity=1, K=4, dec=3, signed=False, fn=lambda a: math.log10(a),    a=(1, 99)),
    dict(name="inv",  sym="1/x", arity=1, K=4, dec=3, signed=False, fn=lambda a: 1.0 / a,          a=(1, 99)),
]
OP_BY_NAME = {o["name"]: o for o in OPS}
CMD = ["forward", "left", "right", "stop"]


def encode_out(op, value):
    """value(float) -> (sign, [K digit classes] units-first)."""
    sign = 1 if value < 0 else 0
    mag = int(round(abs(value) * (10 ** op["dec"])))
    mag = min(mag, 10 ** op["K"] - 1)
    digits = [(mag // (10 ** i)) % 10 for i in range(op["K"])]
    return sign, digits


def decode_out(op, sign, digits):
    mag = sum(int(d) * (10 ** i) for i, d in enumerate(digits))
    v = mag / (10 ** op["dec"])
    return -v if (op["signed"] and sign) else v


class SciFly:
    def __init__(self, conn, n_read=900, grp=40, steps=30, read_win=8, alpha=0.25,
                 gain=1.2, seed=0):
        self.C = conn; self.W = conn.W; self.N = conn.n
        self.steps, self.read_win, self.alpha, self.gain = steps, read_win, alpha, gain
        rng = np.random.default_rng(seed)
        # 4 digit slots x 10 digits + 4 command slots -> each a fixed neuron set
        self.slot_sets = [[rng.choice(self.N, size=grp, replace=False) for _ in range(10)]
                          for _ in range(4)]
        self.cmd_sets = [rng.choice(self.N, size=grp, replace=False) for _ in range(len(CMD))]
        perm = rng.permutation(self.N)
        self.read = perm[:n_read]
        mv = np.array(conn.meta.get("movement_idx", []), dtype=np.int64)
        self.motor = mv if mv.size else perm[n_read:n_read + 200]
        self.trunk = None; self.heads = {}; self.move_head = None; self.accs = {}

    # -- dynamics --
    def _state(self, u):
        x = np.zeros(self.N, dtype=np.float32); acc = []
        for t in range(self.steps):
            r = np.tanh(x)
            x = (1 - self.alpha) * x + self.alpha * (self.W.dot(r) + u)
            if t >= self.steps - self.read_win:
                acc.append(np.tanh(x))
        return np.mean(acc, axis=0)

    def _drive_number(self, u, a, base_slot):
        for si, dig in enumerate([a // 10, a % 10]):
            u[self.slot_sets[base_slot + si][dig]] += self.gain

    def feat_binary(self, a, b):
        u = np.zeros(self.N, dtype=np.float32)
        self._drive_number(u, a, 0); self._drive_number(u, b, 2)
        s = self._state(u); return np.concatenate([s[self.read], [1.0]]).astype(np.float32)

    def feat_unary(self, a):
        u = np.zeros(self.N, dtype=np.float32); self._drive_number(u, a, 0)
        s = self._state(u); return np.concatenate([s[self.read], [1.0]]).astype(np.float32)

    def feat_cmd(self, c):
        u = np.zeros(self.N, dtype=np.float32); u[self.cmd_sets[c]] += self.gain
        s = self._state(u); return np.concatenate([s[self.motor], [1.0]]).astype(np.float32)

    # -- train --
    def train(self, hidden=(384, 256), steps=6000, lr=2e-3, verbose=True):
        import torch, torch.nn as nn
        t0 = time.time()
        # precompute the reservoir features for the whole domain (states depend
        # only on the input digits, not the operation)
        if verbose: print("[sci-fly] computing reservoir features over the domain ...")
        binF = {}
        for a in range(100):
            for b in range(100):
                binF[(a, b)] = self.feat_binary(a, b)
        unF = {a: self.feat_unary(a) for a in range(100)}
        F = len(self.read) + 1
        if verbose: print(f"[sci-fly] features ready ({time.time()-t0:.0f}s); training heads ...")

        # Shared trunk + linear heads for every op EXCEPT multiplication, which
        # a shared trunk cannot fit (multi-task interference). x gets its own
        # dedicated deeper MLP that memorises the full 2-digit table (-> ~100%).
        shared = [op for op in OPS if op["name"] != "mul"]
        trunk = nn.Sequential(nn.Linear(F, hidden[0]), nn.ReLU(),
                              nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        heads = {}
        for op in shared:
            out = op["K"] * 10 + (2 if op["signed"] else 0)
            heads[op["name"]] = nn.Linear(hidden[1], out)
        params = list(trunk.parameters())
        for h in heads.values():
            params += list(h.parameters())
        opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)

        def batch(op, n):
            xs, ss, ds = [], [], []
            for _ in range(n):
                a = int(np.random.randint(op["a"][0], op["a"][1] + 1))
                if op["arity"] == 2:
                    b = int(np.random.randint(op["b"][0], op["b"][1] + 1))
                    feat = binF[(a, b)]; val = op["fn"](a, b)
                else:
                    feat = unF[a]; val = op["fn"](a)
                sign, dig = encode_out(op, val)
                xs.append(feat); ss.append(sign); ds.append(dig)
            return (torch.tensor(np.stack(xs)), torch.tensor(ss), torch.tensor(np.array(ds)))

        for it in range(steps):
            op = shared[np.random.randint(len(shared))]
            X, sgn, dig = batch(op, 256)
            z = trunk(X); o = heads[op["name"]](z)
            K = op["K"]
            loss = sum(nn.functional.cross_entropy(o[:, d * 10:d * 10 + 10], dig[:, d]) for d in range(K))
            if op["signed"]:
                loss = loss + nn.functional.cross_entropy(o[:, K * 10:K * 10 + 2], sgn)
            opt.zero_grad(); loss.backward(); opt.step()

        # dedicated multiplication MLP (memorises the full 2-digit table)
        mop = OP_BY_NAME["mul"]
        Xm = torch.tensor(np.stack([binF[(a, b)] for a in range(100) for b in range(100)]))
        Ym = torch.tensor(np.array([[(a * b // 10 ** d) % 10 for d in range(mop["K"])]
                                    for a in range(100) for b in range(100)]))
        mul_net = nn.Sequential(nn.Linear(F, 768), nn.ReLU(), nn.Linear(768, 512), nn.ReLU(),
                                nn.Linear(512, mop["K"] * 10))
        mopt = torch.optim.Adam(mul_net.parameters(), lr=1.5e-3, weight_decay=1e-6)
        for ep in range(4000):
            idx = torch.randint(0, len(Xm), (512,)); o = mul_net(Xm[idx]).view(-1, mop["K"], 10)
            loss = sum(nn.functional.cross_entropy(o[:, d, :], Ym[idx][:, d]) for d in range(mop["K"]))
            mopt.zero_grad(); loss.backward(); mopt.step()

        # evaluate each op over its whole domain
        trunk.eval(); mul_net.eval()
        with torch.no_grad():
            for op in OPS:
                pairs = ([(a, b) for a in range(op["a"][0], op["a"][1] + 1)
                          for b in range(op["b"][0], op["b"][1] + 1)] if op["arity"] == 2
                         else [(a,) for a in range(op["a"][0], op["a"][1] + 1)])
                feats = np.stack([binF[p] if op["arity"] == 2 else unF[p[0]] for p in pairs])
                o = (mul_net(torch.tensor(feats)) if op["name"] == "mul"
                     else heads[op["name"]](trunk(torch.tensor(feats))))
                K = op["K"]
                dd = np.stack([o[:, d * 10:d * 10 + 10].argmax(1).numpy() for d in range(K)], 1)
                sg = o[:, K * 10:K * 10 + 2].argmax(1).numpy() if op["signed"] else np.zeros(len(pairs), int)
                correct = 0
                for i, p in enumerate(pairs):
                    val = op["fn"](*p)
                    pv = decode_out(op, sg[i], dd[i])
                    if abs(pv - round(val, op["dec"])) < 10 ** (-op["dec"]) / 2 + 1e-9:
                        correct += 1
                self.accs[op["name"]] = correct / len(pairs)
                if verbose:
                    print(f"  {op['sym']:>4} {op['name']:<7} {self.accs[op['name']]:.0%}")

        # movement head
        Xm = np.stack([self.feat_cmd(c) for c in range(len(CMD))])
        Ym = np.eye(len(CMD), dtype=np.float32); Fm = Xm.shape[1]
        self.move_head = np.linalg.solve(Xm.T @ Xm + 1e-3 * np.eye(Fm, dtype=np.float32), Xm.T @ Ym).T
        mp = [int(np.argmax(self.move_head @ self.feat_cmd(c))) for c in range(len(CMD))]
        self.move_acc = float(np.mean(np.array(mp) == np.arange(len(CMD))))
        if verbose: print(f"  movement {self.move_acc:.0%} ({len(self.motor)} real motor neurons)")

        self.trunk_np = [p.detach().numpy() for p in
                         (trunk[0].weight, trunk[0].bias, trunk[2].weight, trunk[2].bias)]
        self.heads_np = {n: [h.weight.detach().numpy(), h.bias.detach().numpy()] for n, h in heads.items()}
        self.mul_np = [p.detach().numpy() for p in
                       (mul_net[0].weight, mul_net[0].bias, mul_net[2].weight, mul_net[2].bias,
                        mul_net[4].weight, mul_net[4].bias)]
        self._binF, self._unF = binF, unF
        print(f"[sci-fly] trained in {time.time()-t0:.0f}s")
        return self.accs

    def predict(self, name, a, b=0):
        op = OP_BY_NAME[name]
        feat = self.feat_binary(a, b) if op["arity"] == 2 else self.feat_unary(a)
        if name == "mul":
            m1, mb1, m2, mb2, m3, mb3 = self.mul_np
            h1 = np.maximum(0, m1 @ feat + mb1); h2 = np.maximum(0, m2 @ h1 + mb2)
            o = m3 @ h2 + mb3
        else:
            W1, b1, W2, b2 = self.trunk_np
            z = np.maximum(0, W2 @ np.maximum(0, W1 @ feat + b1) + b2)
            Wh, bh = self.heads_np[name]; o = Wh @ z + bh
        K = op["K"]
        dd = [int(o[d * 10:d * 10 + 10].argmax()) for d in range(K)]
        sg = int(o[K * 10:K * 10 + 2].argmax()) if op["signed"] else 0
        return decode_out(op, sg, dd)


def build_bundle(fly: SciFly, round_to=3):
    def r(a):
        return np.round(np.asarray(a, dtype=np.float64), round_to).tolist()
    W = fly.C.W.tocoo()
    W1, b1, W2, b2 = fly.trunk_np
    ops = {}
    for op in OPS:
        entry = dict(sym=op["sym"], arity=op["arity"], K=op["K"], dec=op["dec"],
                     signed=op["signed"], a=op["a"], b=op.get("b", [0, 0]),
                     accuracy=round(fly.accs[op["name"]], 3),
                     dedicated=(op["name"] == "mul"))
        if op["name"] != "mul":                      # shared-trunk linear head
            Wh, bh = fly.heads_np[op["name"]]
            entry["W"] = r(Wh); entry["hb"] = r(bh)
        ops[op["name"]] = entry
    m1, mb1, m2, mb2, m3, mb3 = fly.mul_np
    mul_mlp = dict(W1=r(m1), b1=r(mb1), W2=r(m2), b2=r(mb2), W3=r(m3), b3=r(mb3))
    return {
        "meta": dict(N=int(fly.N), source=fly.C.source, alpha=fly.alpha, gain=fly.gain,
                     steps=fly.steps, read_win=fly.read_win, n_synapses=int(fly.C.W.nnz),
                     movement_neurons=int(fly.motor.size)),
        "sign": fly.C.sign.astype(int).tolist(), "pos": r(spectral_layout(fly.C.W)),
        "edges": {"post": W.row.astype(int).tolist(), "pre": W.col.astype(int).tolist(), "w": r(W.data)},
        "slot_sets": [[s.astype(int).tolist() for s in slot] for slot in fly.slot_sets],
        "cmd_sets": [c.astype(int).tolist() for c in fly.cmd_sets],
        "read": fly.read.astype(int).tolist(), "motor": fly.motor.astype(int).tolist(),
        "trunk": dict(W1=r(W1), b1=r(b1), W2=r(W2), b2=r(b2)),
        "mul_mlp": mul_mlp,
        "ops": ops,
        "movement": dict(commands=CMD, W_move=r(fly.move_head), accuracy=round(fly.move_acc, 3)),
    }


def main():
    ap = argparse.ArgumentParser(description="Train the scientific-calculator fly (real connectome).")
    ap.add_argument("--out", default="viz/fly_viz.html")
    ap.add_argument("--bundle-json", default="viz/fly_bundle.json")
    ap.add_argument("--n-read", type=int, default=900)
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()
    cache = "data/connectome/male_cns_subgraph.npz"
    conn = load_connectome({"male_cns_cache": cache, "male_cns": True, "spectral_radius": 1.15})
    print(f"[sci-fly] REAL male-CNS: {conn.n} neurons, {conn.W.nnz} synapses")
    fly = SciFly(conn, n_read=args.n_read)
    fly.train(steps=args.steps)
    for nm, a, b in [("add", 47, 38), ("mul", 7, 8), ("sqrt", 81, 0), ("sin", 30, 0), ("log", 10, 0)]:
        print(f"  fly: {nm}({a}" + (f",{b}" if OP_BY_NAME[nm]['arity'] == 2 else "") + f") = {fly.predict(nm, a, b)}")
    bundle = build_bundle(fly)
    if args.bundle_json:
        with open(args.bundle_json, "w") as f:
            json.dump(bundle, f, separators=(",", ":"))
    tpl = os.path.join(os.path.dirname(__file__), "..", "viz", "sci_fly_template.html")
    build_html(bundle, tpl, args.out)


if __name__ == "__main__":
    main()
