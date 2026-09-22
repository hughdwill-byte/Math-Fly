"""
The scientific-calculator fly: the REAL male-CNS connectome trained to behave
like a pocket scientific calculator, driven by an expression evaluator.

Design (all measured, not assumed):
  * numbers are 0..999, encoded as three sustained one-hot digit slots each;
  * + and - GENERALISE (~98% on held-out 3-digit pairs), so they're trained on
    sampled pairs over the whole 0..999 range with a dedicated MLP;
  * x and / do NOT generalise (a shared/linear readout can't, and 3-digit tables
    are too large to memorise), so they stay EXACT over 2-digit operands (0..99)
    with a dedicated MLP that memorises the full table;
  * the unary functions (x^2, sqrt, sin, cos, log10, 1/x) are exact over 0..99
    from a shared trunk with linear heads;
  * movement is read from the fly's real motor neurons.

Each operation reports its real accuracy and its valid operand range, and the
visualiser's expression evaluator (shunting-yard, with precedence) calls the fly
for every elementary step -- so `2 + 3 x 4` = 14, computed by the real fly.
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

# head kinds: "gen" (dedicated MLP, generalises), "memo" (dedicated deeper MLP,
# memorises a full table), "trunk" (shared trunk + linear head)
OPS = [
    dict(name="add", sym="+", arity=2, K=4, dec=0, signed=False, kind="gen",  a=(0, 999), b=(0, 999), fn=lambda a, b: a + b),
    dict(name="sub", sym="−", arity=2, K=3, dec=0, signed=True, kind="gen", a=(0, 999), b=(0, 999), fn=lambda a, b: a - b),
    dict(name="mul", sym="×", arity=2, K=4, dec=0, signed=False, kind="memo", a=(0, 99), b=(0, 99), fn=lambda a, b: a * b),
    dict(name="div", sym="÷", arity=2, K=2, dec=0, signed=False, kind="memo", a=(0, 99), b=(1, 99), fn=lambda a, b: a // b),
    dict(name="square", sym="x²", arity=1, K=4, dec=0, signed=False, kind="trunk", a=(0, 99), fn=lambda a: a * a),
    dict(name="sqrt", sym="√", arity=1, K=3, dec=2, signed=False, kind="trunk", a=(0, 99), fn=lambda a: math.sqrt(a)),
    dict(name="sin", sym="sin", arity=1, K=4, dec=3, signed=True, kind="trunk", a=(0, 99), fn=lambda a: math.sin(math.radians(a))),
    dict(name="cos", sym="cos", arity=1, K=4, dec=3, signed=True, kind="trunk", a=(0, 99), fn=lambda a: math.cos(math.radians(a))),
    dict(name="log", sym="log", arity=1, K=4, dec=3, signed=False, kind="trunk", a=(1, 99), fn=lambda a: math.log10(a)),
    dict(name="inv", sym="1/x", arity=1, K=4, dec=3, signed=False, kind="trunk", a=(1, 99), fn=lambda a: 1.0 / a),
]
OP_BY_NAME = {o["name"]: o for o in OPS}
CMD = ["forward", "left", "right", "stop"]
NSLOT = 6          # 3 digit slots per operand, two operands
DPN = 3            # digits per number (0..999)


def encode_out(op, value):
    sign = 1 if value < 0 else 0
    mag = min(int(round(abs(value) * (10 ** op["dec"]))), 10 ** op["K"] - 1)
    return sign, [(mag // (10 ** i)) % 10 for i in range(op["K"])]


def decode_out(op, sign, digits):
    mag = sum(int(d) * (10 ** i) for i, d in enumerate(digits))
    v = mag / (10 ** op["dec"])
    return -v if (op["signed"] and sign) else v


def _digs(x):
    return [x // 100, (x // 10) % 10, x % 10]


class SciFly:
    def __init__(self, conn, n_read=600, grp=34, steps=30, read_win=8, alpha=0.25,
                 gain=1.2, seed=0):
        self.C = conn; self.W = conn.W; self.N = conn.n
        self.steps, self.read_win, self.alpha, self.gain = steps, read_win, alpha, gain
        rng = np.random.default_rng(seed)
        self.slot_sets = [[rng.choice(self.N, size=grp, replace=False) for _ in range(10)]
                          for _ in range(NSLOT)]
        self.cmd_sets = [rng.choice(self.N, size=grp, replace=False) for _ in range(len(CMD))]
        perm = rng.permutation(self.N)
        self.read = perm[:n_read]
        mv = np.array(conn.meta.get("movement_idx", []), dtype=np.int64)
        self.motor = mv if mv.size else perm[n_read:n_read + 200]

    def _state(self, u):
        x = np.zeros(self.N, dtype=np.float32); acc = []
        for t in range(self.steps):
            r = np.tanh(x)
            x = (1 - self.alpha) * x + self.alpha * (self.W.dot(r) + u)
            if t >= self.steps - self.read_win:
                acc.append(np.tanh(x))
        return np.mean(acc, axis=0)

    def _num(self, u, x, base):
        for si, dg in enumerate(_digs(x)):
            u[self.slot_sets[base + si][dg]] += self.gain

    def feat_bin(self, a, b):
        u = np.zeros(self.N, dtype=np.float32); self._num(u, a, 0); self._num(u, b, 3)
        return np.concatenate([self._state(u)[self.read], [1.0]]).astype(np.float32)

    def feat_un(self, a):
        u = np.zeros(self.N, dtype=np.float32); self._num(u, a, 0)
        return np.concatenate([self._state(u)[self.read], [1.0]]).astype(np.float32)

    def feat_cmd(self, c):
        u = np.zeros(self.N, dtype=np.float32); u[self.cmd_sets[c]] += self.gain
        return np.concatenate([self._state(u)[self.motor], [1.0]]).astype(np.float32)

    def train(self, verbose=True):
        import torch, torch.nn as nn
        t0 = time.time()
        F = len(self.read) + 1
        if verbose: print("[sci-fly] computing reservoir features ...")
        # feature banks
        gen_pairs = [(int(np.random.randint(0, 1000)), int(np.random.randint(0, 1000))) for _ in range(22000)]
        genF = {}
        for ab in gen_pairs:
            if ab not in genF:
                genF[ab] = self.feat_bin(*ab)
        memoF = {(a, b): self.feat_bin(a, b) for a in range(100) for b in range(100)}
        unF = {a: self.feat_un(a) for a in range(100)}
        if verbose: print(f"[sci-fly] features ready ({time.time()-t0:.0f}s); training ...")

        self.heads = {}; self.trunk_np = None
        # ---- shared trunk + linear heads for the unary (0..99) ops ----
        trunk_ops = [op for op in OPS if op["kind"] == "trunk"]
        trunk = nn.Sequential(nn.Linear(F, 384), nn.ReLU(), nn.Linear(384, 256), nn.ReLU())
        lin = {op["name"]: nn.Linear(256, op["K"] * 10 + (2 if op["signed"] else 0)) for op in trunk_ops}
        p = list(trunk.parameters())
        for h in lin.values():
            p += list(h.parameters())
        opt = torch.optim.Adam(p, lr=2e-3, weight_decay=1e-5)
        X_un = torch.tensor(np.stack([unF[a] for a in range(100)]))
        for it in range(4000):
            op = trunk_ops[np.random.randint(len(trunk_ops))]
            dom = list(range(op["a"][0], op["a"][1] + 1))
            idx = np.random.choice(dom, 128)
            X = X_un[idx]; o = lin[op["name"]](trunk(X)); K = op["K"]
            tgt = [encode_out(op, op["fn"](a)) for a in idx]
            dig = torch.tensor(np.array([t[1] for t in tgt])); sgn = torch.tensor(np.array([t[0] for t in tgt]))
            loss = sum(nn.functional.cross_entropy(o[:, d * 10:d * 10 + 10], dig[:, d]) for d in range(K))
            if op["signed"]:
                loss = loss + nn.functional.cross_entropy(o[:, K * 10:K * 10 + 2], sgn)
            opt.zero_grad(); loss.backward(); opt.step()
        self.trunk_np = [q.detach().numpy() for q in (trunk[0].weight, trunk[0].bias, trunk[2].weight, trunk[2].bias)]
        self._trunk = trunk; self._lin = lin

        # ---- dedicated MLPs for the binary ops ----
        def train_mlp(op, feats_map, pairs, deep, iters):
            hidden = (768, 512) if deep else (512, 256)
            layers = [nn.Linear(F, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                      nn.Linear(hidden[1], op["K"] * 10 + (2 if op["signed"] else 0))]
            net = nn.Sequential(*layers)
            o2 = torch.optim.Adam(net.parameters(), lr=1.5e-3, weight_decay=1e-6)
            X = torch.tensor(np.stack([feats_map[ab] for ab in pairs]))
            outs = [encode_out(op, op["fn"](*ab)) for ab in pairs]
            dig = torch.tensor(np.array([o[1] for o in outs])); sgn = torch.tensor(np.array([o[0] for o in outs]))
            K = op["K"]
            for it in range(iters):
                idx = torch.randint(0, len(X), (512,)); o = net(X[idx])
                loss = sum(nn.functional.cross_entropy(o[:, d * 10:d * 10 + 10], dig[idx][:, d]) for d in range(K))
                if op["signed"]:
                    loss = loss + nn.functional.cross_entropy(o[:, K * 10:K * 10 + 2], sgn[idx])
                o2.zero_grad(); loss.backward(); o2.step()
            return net

        nets = {}
        gen_list = list(genF.keys())
        for op in OPS:
            if op["kind"] == "gen":
                nets[op["name"]] = train_mlp(op, genF, gen_list, deep=False, iters=1600)
            elif op["kind"] == "memo":
                pairs = [(a, b) for a in range(op["a"][0], op["a"][1] + 1) for b in range(op["b"][0], op["b"][1] + 1)]
                nets[op["name"]] = train_mlp(op, memoF, pairs, deep=True, iters=4000)
        self._nets = nets
        self.mlp_np = {n: [q.detach().numpy() for q in (net[0].weight, net[0].bias, net[2].weight, net[2].bias, net[4].weight, net[4].bias)]
                       for n, net in nets.items()}

        # ---- evaluate every op ----
        self.accs = {}
        with torch.no_grad():
            for op in OPS:
                if op["arity"] == 2:
                    if op["kind"] == "gen":
                        pairs = [(int(np.random.randint(0, 1000)), int(np.random.randint(0, 1000))) for _ in range(2000)]
                        feats = np.stack([self.feat_bin(*ab) for ab in pairs])
                    else:
                        pairs = [(a, b) for a in range(op["a"][0], op["a"][1] + 1) for b in range(op["b"][0], op["b"][1] + 1)]
                        feats = np.stack([memoF[ab] for ab in pairs])
                    o = nets[op["name"]](torch.tensor(feats))
                else:
                    pairs = [(a,) for a in range(op["a"][0], op["a"][1] + 1)]
                    feats = np.stack([unF[p[0]] for p in pairs])
                    o = lin[op["name"]](trunk(torch.tensor(feats)))
                K = op["K"]
                dd = np.stack([o[:, d * 10:d * 10 + 10].argmax(1).numpy() for d in range(K)], 1)
                sg = o[:, K * 10:K * 10 + 2].argmax(1).numpy() if op["signed"] else np.zeros(len(pairs), int)
                c = sum(abs(decode_out(op, sg[i], dd[i]) - round(op["fn"](*p), op["dec"])) < 10 ** (-op["dec"]) / 2 + 1e-9
                        for i, p in enumerate(pairs))
                self.accs[op["name"]] = c / len(pairs)
                if verbose:
                    print(f"  {op['sym']:>4} {op['name']:<7} {self.accs[op['name']]:.0%}  ({op['a'][0]}-{op['a'][1]})")

        # movement
        Xm = np.stack([self.feat_cmd(c) for c in range(len(CMD))]); Fm = Xm.shape[1]
        self.move_head = np.linalg.solve(Xm.T @ Xm + 1e-3 * np.eye(Fm, dtype=np.float32), Xm.T @ np.eye(len(CMD), dtype=np.float32)).T
        mp = [int(np.argmax(self.move_head @ self.feat_cmd(c))) for c in range(len(CMD))]
        self.move_acc = float(np.mean(np.array(mp) == np.arange(len(CMD))))
        if verbose: print(f"  movement {self.move_acc:.0%} ({len(self.motor)} real motor neurons)")
        print(f"[sci-fly] trained in {time.time()-t0:.0f}s")
        return self.accs

    def predict(self, name, a, b=0):
        op = OP_BY_NAME[name]
        feat = self.feat_bin(a, b) if op["arity"] == 2 else self.feat_un(a)
        if op["kind"] == "trunk":
            W1, b1, W2, b2 = self.trunk_np
            z = np.maximum(0, W2 @ np.maximum(0, W1 @ feat + b1) + b2)
            L = self._lin[name]; o = L.weight.detach().numpy() @ z + L.bias.detach().numpy()
        else:
            m1, mb1, m2, mb2, m3, mb3 = self.mlp_np[name]
            h1 = np.maximum(0, m1 @ feat + mb1); h2 = np.maximum(0, m2 @ h1 + mb2); o = m3 @ h2 + mb3
        K = op["K"]
        dd = [int(o[d * 10:d * 10 + 10].argmax()) for d in range(K)]
        sg = int(o[K * 10:K * 10 + 2].argmax()) if op["signed"] else 0
        return decode_out(op, sg, dd)


def build_bundle(fly, round_to=3):
    def r(a):
        return np.round(np.asarray(a, dtype=np.float64), round_to).tolist()
    W = fly.C.W.tocoo(); W1, b1, W2, b2 = fly.trunk_np
    ops = {}
    for op in OPS:
        e = dict(sym=op["sym"], arity=op["arity"], K=op["K"], dec=op["dec"], signed=op["signed"],
                 kind=op["kind"], a=op["a"], b=op.get("b", [0, 0]), accuracy=round(fly.accs[op["name"]], 3))
        if op["kind"] == "trunk":
            L = fly._lin[op["name"]]
            e["W"] = r(L.weight.detach().numpy()); e["hb"] = r(L.bias.detach().numpy())
        else:
            m = fly.mlp_np[op["name"]]
            e["mlp"] = dict(W1=r(m[0]), b1=r(m[1]), W2=r(m[2]), b2=r(m[3]), W3=r(m[4]), b3=r(m[5]))
        ops[op["name"]] = e
    return {
        "meta": dict(N=int(fly.N), source=fly.C.source, alpha=fly.alpha, gain=fly.gain,
                     steps=fly.steps, read_win=fly.read_win, n_synapses=int(fly.C.W.nnz),
                     movement_neurons=int(fly.motor.size), digits_per_num=DPN),
        "sign": fly.C.sign.astype(int).tolist(), "pos": r(spectral_layout(fly.C.W)),
        "edges": {"post": W.row.astype(int).tolist(), "pre": W.col.astype(int).tolist(), "w": r(W.data)},
        "slot_sets": [[s.astype(int).tolist() for s in slot] for slot in fly.slot_sets],
        "cmd_sets": [c.astype(int).tolist() for c in fly.cmd_sets],
        "read": fly.read.astype(int).tolist(), "motor": fly.motor.astype(int).tolist(),
        "trunk": dict(W1=r(W1), b1=r(b1), W2=r(W2), b2=r(b2)),
        "ops": ops,
        "movement": dict(commands=CMD, W_move=r(fly.move_head), accuracy=round(fly.move_acc, 3)),
    }


def main():
    ap = argparse.ArgumentParser(description="Train the scientific-calculator fly (real connectome).")
    ap.add_argument("--out", default="viz/fly_viz.html")
    ap.add_argument("--bundle-json", default="viz/fly_bundle.json")
    ap.add_argument("--n-read", type=int, default=600)
    args = ap.parse_args()
    conn = load_connectome({"male_cns_cache": "data/connectome/male_cns_subgraph.npz",
                            "male_cns": True, "spectral_radius": 1.15})
    print(f"[sci-fly] REAL male-CNS: {conn.n} neurons, {conn.W.nnz} synapses")
    fly = SciFly(conn, n_read=args.n_read)
    fly.train()
    for nm, a, b in [("add", 471, 380), ("sub", 900, 250), ("mul", 7, 8), ("div", 84, 7),
                     ("sqrt", 81, 0), ("sin", 30, 0), ("log", 10, 0)]:
        print(f"  fly: {nm}({a}" + (f",{b}" if OP_BY_NAME[nm]['arity'] == 2 else "") + f") = {fly.predict(nm, a, b)}")
    bundle = build_bundle(fly)
    if args.bundle_json:
        with open(args.bundle_json, "w") as f:
            json.dump(bundle, f, separators=(",", ":"))
    tpl = os.path.join(os.path.dirname(__file__), "..", "viz", "sci_fly_template.html")
    build_html(bundle, tpl, args.out)


if __name__ == "__main__":
    main()
