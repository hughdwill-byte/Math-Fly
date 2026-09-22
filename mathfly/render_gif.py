"""
Render an animated GIF of the fly solving a problem -- the same real dynamics
the interactive page shows, baked into a shareable loop for READMEs / slides.

It replays the trained model's genuine state trajectory (via export_web's
reference_forward), drawing neurons coloured by firing rate, signal sparks
travelling along synapses, the incoming token stream, and the answer resolving.

Usage:
    # uses viz/fly_bundle.json if present, else trains a fresh model
    python -m mathfly.render_gif --op + --a 47 --b 38 --out viz/fly_solve.gif
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .export_web import reference_forward, train_for_viz, build_bundle

BG = (7, 11, 20)
EXC = (55, 224, 198)
INH = (255, 95, 141)
GOLD = (255, 206, 84)
INK = (232, 238, 249)
MUTED = (139, 160, 194)


def _font(size):
    for name in ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _active_edges(bundle, R, frame, k=180):
    pre = np.array(bundle["edges"]["pre"]); post = np.array(bundle["edges"]["post"])
    w = np.array(bundle["edges"]["w"], dtype=np.float32)
    # strongest synapses only, gated by presynaptic activity
    strong = np.argsort(-np.abs(w))[:3000]
    src = R[frame]
    st = np.abs(src[pre[strong]]) * np.minimum(4.0, np.abs(w[strong]))
    keep = strong[st > 0.35]
    order = np.argsort(-(np.abs(src[pre[keep]]) * np.minimum(4.0, np.abs(w[keep]))))[:k]
    idx = keep[order]
    return pre[idx], post[idx], (w[idx] < 0)


def render_gif(bundle, op, a, b, out="viz/fly_solve.gif", W=560, H=396, scale=2,
               fps=12, hold=8, colors=64):
    answer, R, U, tokens, (win_start, T) = reference_forward(bundle, op, a, b)
    truth = {"+": a + b, "-": abs(a - b), "*": a * b, ">": int(a > b)}[op]
    disp_op = {"+": "+", "-": "−", "*": "×", ">": ">"}[op]

    pos = np.array(bundle["pos"], dtype=np.float32)
    sign = np.array(bundle["sign"])
    IN = np.array(bundle["input_neurons"])
    edges_pre = np.array(bundle["edges"]["pre"]); edges_post = np.array(bundle["edges"]["post"])
    e_w = np.array(bundle["edges"]["w"], dtype=np.float32)

    Ws, Hs = W * scale, H * scale
    padX, padY = 30 * scale, 74 * scale
    XY = np.empty((len(pos), 2))
    XY[:, 0] = padX + pos[:, 0] * (Ws - 2 * padX)
    XY[:, 1] = padY + pos[:, 1] * (Hs - 2 * padY)

    fbig = _font(30 * scale); fsmall = _font(15 * scale); ftok = _font(20 * scale)

    # static faint edge layer
    base = Image.new("RGB", (Ws, Hs), BG)
    ed = ImageDraw.Draw(base)
    stride = max(1, len(e_w) // 2500)   # sparser web keeps the GIF light
    for k in range(0, len(e_w), stride):
        A = XY[edges_pre[k]]; B = XY[edges_post[k]]
        col = INH if e_w[k] < 0 else EXC
        ed.line([tuple(A), tuple(B)], fill=_lerp(BG, col, 0.10), width=1)

    frames = []
    seq = list(range(T)) + [T - 1] * hold          # hold on the answer at the end
    for fi, t in enumerate(seq):
        img = base.copy(); d = ImageDraw.Draw(img, "RGBA")
        # sparks
        pre, post, inh = _active_edges(bundle, R, t)
        frac = (fi * 0.22) % 1.0
        for j in range(len(pre)):
            A = XY[pre[j]]; B = XY[post[j]]; col = INH if inh[j] else EXC
            x = A[0] + (B[0] - A[0]) * frac; y = A[1] + (B[1] - A[1]) * frac
            d.line([tuple(A), tuple(B)], fill=col + (26,), width=1)
            rr = int(2.2 * scale)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=col + (220,))
        # neurons
        rt = R[t]
        for i in range(len(pos)):
            act = abs(float(rt[i]))
            col = EXC if sign[i] >= 0 else INH
            rad = (1.5 + act * 3.8) * scale
            alpha = int((0.10 + act * 0.85) * 255)
            x, y = XY[i]
            if act > 0.06:
                gr = rad * 2.2
                d.ellipse([x - gr, y - gr, x + gr, y + gr], fill=col + (int(alpha * 0.25),))
            d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=col + (alpha,))
        # sensory rings
        for i in IN:
            x, y = XY[i]; rr = 2.6 * scale
            d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=GOLD + (110,), width=1)

        # header: equation + forming answer
        in_settle = t >= win_start
        if in_settle:
            heads = bundle["tasks"][op]
            feat = np.concatenate([R[win_start:t + 1][:, np.array(bundle["readout_neurons"])].mean(0), [1.0]]).astype(np.float32)
            if heads["mode"] == "digits":
                dd = np.argmax(np.array(heads["digit_heads"], dtype=np.float32) @ feat, axis=1)
                cur = int(sum(int(v) * (10 ** i) for i, v in enumerate(dd)))
                cur = str(cur)
            else:
                cur = "TRUE" if np.argmax(np.array(heads["W_out"], dtype=np.float32) @ feat) == 1 else "FALSE"
        else:
            cur = "?"
        eq = f"{a} {disp_op} {b} ="
        d.text((30 * scale, 20 * scale), eq, font=fbig, fill=INK)
        w_eq = d.textlength(eq, font=fbig)
        ans_col = GOLD if (fi < len(seq) - hold or answer == truth) else (255, 107, 107)
        d.text((30 * scale + w_eq + 14 * scale, 20 * scale), cur, font=fbig, fill=ans_col)
        # phase label
        phase = "thinking . . ." if in_settle else "reading input"
        d.text((30 * scale, 58 * scale), phase, font=fsmall, fill=MUTED)
        # final verdict
        if t == T - 1 and fi >= len(seq) - hold:
            vt = "correct" if answer == truth else f"guessed (true {truth if op!='>' else ('TRUE' if truth else 'FALSE')})"
            vc = EXC if answer == truth else (255, 107, 107)
            d.text((Ws - 30 * scale - d.textlength(vt, font=fsmall), 58 * scale), vt, font=fsmall, fill=vc)

        frames.append(img.resize((W, H), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=colors))

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], loop=0,
                   duration=int(1000 / fps), optimize=True, disposal=2)
    print(f"[gif] wrote {out} ({os.path.getsize(out)/1e6:.2f} MB, {len(frames)} frames)  "
          f"fly said {a}{op}{b} = {answer} ({'correct' if answer==truth else 'true '+str(truth)})")
    return out


def render_calc_gif(bundle, op, a, b, out="viz/fly_solve.gif", W=560, H=396,
                    scale=2, fps=12, hold=8, colors=64):
    """Render the calculator-fly (sustained one-hot) solving one problem."""
    m = bundle["meta"]
    N = m["N"]; alpha = m["alpha"]; gain = m["gain"]; steps = m["steps"]; rwin = m["read_win"]
    pos = np.array(bundle["pos"], dtype=np.float32); sign = np.array(bundle["sign"])
    groups = [np.array(g) for g in bundle["groups"]]
    read = np.array(bundle["read"]); motor = np.array(bundle["motor"])
    e_pre = np.array(bundle["edges"]["pre"]); e_post = np.array(bundle["edges"]["post"])
    e_w = np.array(bundle["edges"]["w"], dtype=np.float32)
    bias = np.array(bundle.get("bias", np.zeros(N)), dtype=np.float32)
    if op == "-" and b > a:
        a, b = b, a
    u = np.zeros(N, dtype=np.float32)
    for n in groups[a]: u[n] += gain
    for n in groups[10 + b]: u[n] += gain
    x = np.zeros(N, dtype=np.float32); R = []
    for t in range(steps):
        r = np.tanh(x); rec = np.zeros(N, dtype=np.float32); np.add.at(rec, e_post, e_w * r[e_pre])
        x = (1 - alpha) * x + alpha * (rec + u + bias); R.append(np.tanh(x))
    R = np.array(R)
    feat = np.concatenate([R[steps - rwin:][:, read].mean(0), [1.0]]).astype(np.float32)
    t = bundle["tasks"][op]; mlp = t["mlp"]
    h = np.maximum(0, np.array(mlp["W1"]) @ feat + np.array(mlp["b1"]))
    o = np.array(mlp["W2"]) @ h + np.array(mlp["b2"])
    if t["compare"]:
        ans = int(np.argmax(o)); truth = int(a > b)
        shown = "TRUE" if ans else "FALSE"; tshown = "TRUE" if truth else "FALSE"
    else:
        nd = m["n_digits"]; dd = o.reshape(nd, 10).argmax(1)
        ans = int((dd * (10 ** np.arange(nd))).sum())
        truth = {"+": a + b, "-": a - b, "*": a * b}[op]; shown = str(ans); tshown = str(truth)
    disp_op = {"+": "+", "-": "−", "*": "×", ">": ">"}[op]

    Ws, Hs = W * scale, H * scale; padX, padY = 30 * scale, 74 * scale
    XY = np.empty((N, 2)); XY[:, 0] = padX + pos[:, 0] * (Ws - 2 * padX); XY[:, 1] = padY + pos[:, 1] * (Hs - 2 * padY)
    fbig = _font(30 * scale); fsmall = _font(15 * scale)
    base = Image.new("RGB", (Ws, Hs), BG); ed = ImageDraw.Draw(base)
    stride = max(1, len(e_w) // 2500)
    for k in range(0, len(e_w), stride):
        A = XY[e_pre[k]]; Bp = XY[e_post[k]]; ed.line([tuple(A), tuple(Bp)], fill=_lerp(BG, INH if e_w[k] < 0 else EXC, 0.09), width=1)
    in_neurons = set(groups[a].tolist()) | set(groups[10 + b].tolist())
    frames = []
    seq = list(range(steps)) + [steps - 1] * hold
    for fi, tt in enumerate(seq):
        img = base.copy(); d = ImageDraw.Draw(img, "RGBA"); rt = R[tt]
        for i in range(N):
            act = abs(float(rt[i])); col = EXC if sign[i] >= 0 else INH
            rad = (1.4 + act * 3.6) * scale; al = int((0.09 + act * 0.85) * 255); xx, yy = XY[i]
            if act > 0.06:
                gr = rad * 2.1; d.ellipse([xx - gr, yy - gr, xx + gr, yy + gr], fill=col + (int(al * 0.25),))
            d.ellipse([xx - rad, yy - rad, xx + rad, yy + rad], fill=col + (al,))
        for i in in_neurons:
            xx, yy = XY[i]; rr = 3.0 * scale; d.ellipse([xx - rr, yy - rr, xx + rr, yy + rr], outline=GOLD + (150,), width=2)
        for i in motor:
            xx, yy = XY[i]; rr = 2.2 * scale; d.ellipse([xx - rr, yy - rr, xx + rr, yy + rr], outline=(138, 180, 255, 90), width=1)
        in_read = tt >= steps - rwin
        cur = shown if in_read else "?"
        eq = f"{a} {disp_op} {b} ="; d.text((30 * scale, 20 * scale), eq, font=fbig, fill=INK)
        weq = d.textlength(eq, font=fbig)
        col_ans = GOLD if (fi < len(seq) - hold or ans == truth) else (255, 107, 107)
        d.text((30 * scale + weq + 14 * scale, 20 * scale), cur, font=fbig, fill=col_ans)
        d.text((30 * scale, 58 * scale), "holding both numbers" if not in_read else "computing . . .", font=fsmall, fill=MUTED)
        if tt == steps - 1 and fi >= len(seq) - hold:
            vt = "correct" if ans == truth else f"true {tshown}"; vc = EXC if ans == truth else (255, 107, 107)
            d.text((Ws - 30 * scale - d.textlength(vt, font=fsmall), 58 * scale), vt, font=fsmall, fill=vc)
        frames.append(img.resize((W, H), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=colors))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], loop=0, duration=int(1000 / fps), optimize=True, disposal=2)
    print(f"[gif] wrote {out} ({os.path.getsize(out)/1e6:.2f} MB)  real fly: {a}{op}{b} = {shown} ({'correct' if ans==truth else 'true '+tshown})")
    return out


def render_sci_gif(bundle, name, a, b=0, out="viz/fly_solve.gif", W=560, H=396,
                   scale=2, fps=12, hold=8, colors=64):
    """Render the scientific-calculator fly solving one problem."""
    m = bundle["meta"]; N = m["N"]; alpha = m["alpha"]; gain = m["gain"]; steps = m["steps"]; rwin = m["read_win"]
    pos = np.array(bundle["pos"], dtype=np.float32); sign = np.array(bundle["sign"])
    slots = [[np.array(s) for s in slot] for slot in bundle["slot_sets"]]
    read = np.array(bundle["read"]); motor = np.array(bundle["motor"])
    e_pre = np.array(bundle["edges"]["pre"]); e_post = np.array(bundle["edges"]["post"]); e_w = np.array(bundle["edges"]["w"], dtype=np.float32)
    op = bundle["ops"][name]; tr = bundle["trunk"]
    def digs(v):
        return [v // 100, (v // 10) % 10, v % 10]
    u = np.zeros(N, dtype=np.float32); in_neurons = set()
    for si, dg in enumerate(digs(a)):
        u[slots[si][dg]] += gain; in_neurons |= set(slots[si][dg].tolist())
    if op["arity"] == 2:
        for si, dg in enumerate(digs(b)):
            u[slots[3 + si][dg]] += gain; in_neurons |= set(slots[3 + si][dg].tolist())
    x = np.zeros(N, dtype=np.float32); R = []
    for t in range(steps):
        r = np.tanh(x); rec = np.zeros(N, dtype=np.float32); np.add.at(rec, e_post, e_w * r[e_pre])
        x = (1 - alpha) * x + alpha * (rec + u); R.append(np.tanh(x))
    R = np.array(R)
    feat = np.concatenate([R[steps - rwin:][:, read].mean(0), [1.0]]).astype(np.float32)
    if op["kind"] == "trunk":
        z = np.maximum(0, np.array(tr["W2"]) @ np.maximum(0, np.array(tr["W1"]) @ feat + np.array(tr["b1"])) + np.array(tr["b2"]))
        o = np.array(op["W"]) @ z + np.array(op["hb"])
    else:
        mm = op["mlp"]
        h1 = np.maximum(0, np.array(mm["W1"]) @ feat + np.array(mm["b1"]))
        h2 = np.maximum(0, np.array(mm["W2"]) @ h1 + np.array(mm["b2"]))
        o = np.array(mm["W3"]) @ h2 + np.array(mm["b3"])
    K = op["K"]; mag = sum(int(o[d * 10:d * 10 + 10].argmax()) * (10 ** i) for i, d in enumerate(range(K)))
    val = mag / (10 ** op["dec"])
    if op["signed"] and o[K * 10 + 1] > o[K * 10]:
        val = -val
    shown = f"{val:.{op['dec']}f}" if op["dec"] > 0 else str(int(round(val)))
    disp = f"{a} {op['sym']} {b} =" if op["arity"] == 2 else f"{op['sym']}({a}) ="

    Ws, Hs = W * scale, H * scale
    # letterbox the anatomical layout (preserve aspect), leaving room for the header
    x0, x1, y0, y1 = pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(), pos[:, 1].max()
    topPad = 84 * scale; s = min((Ws - 60 * scale) / (x1 - x0), (Hs - topPad - 30 * scale) / (y1 - y0))
    ox = (Ws - (x1 - x0) * s) / 2; oy = topPad
    XY = np.empty((N, 2)); XY[:, 0] = ox + (pos[:, 0] - x0) * s; XY[:, 1] = oy + (pos[:, 1] - y0) * s
    fbig = _font(28 * scale); fsmall = _font(15 * scale)
    base = Image.new("RGB", (Ws, Hs), BG); ed = ImageDraw.Draw(base)
    stride = max(1, len(e_w) // 2500)
    for k in range(0, len(e_w), stride):
        A = XY[e_pre[k]]; Bp = XY[e_post[k]]; ed.line([tuple(A), tuple(Bp)], fill=_lerp(BG, INH if e_w[k] < 0 else EXC, 0.09), width=1)
    frames = []; seq = list(range(steps)) + [steps - 1] * hold
    for fi, tt in enumerate(seq):
        img = base.copy(); d = ImageDraw.Draw(img, "RGBA"); rt = R[tt]
        for i in range(N):
            act = abs(float(rt[i])); col = EXC if sign[i] >= 0 else INH
            rad = (1.4 + act * 3.6) * scale; al = int((0.09 + act * 0.85) * 255); xx, yy = XY[i]
            if act > 0.06:
                gr = rad * 2.1; d.ellipse([xx - gr, yy - gr, xx + gr, yy + gr], fill=col + (int(al * 0.25),))
            d.ellipse([xx - rad, yy - rad, xx + rad, yy + rad], fill=col + (al,))
        for i in in_neurons:
            xx, yy = XY[i]; rr = 3.0 * scale; d.ellipse([xx - rr, yy - rr, xx + rr, yy + rr], outline=GOLD + (150,), width=2)
        for i in motor:
            xx, yy = XY[i]; rr = 2.2 * scale; d.ellipse([xx - rr, yy - rr, xx + rr, yy + rr], outline=(138, 180, 255, 90), width=1)
        in_read = tt >= steps - rwin; cur = shown if in_read else "?"
        d.text((30 * scale, 20 * scale), disp, font=fbig, fill=INK); weq = d.textlength(disp, font=fbig)
        d.text((30 * scale + weq + 14 * scale, 20 * scale), cur, font=fbig, fill=GOLD)
        d.text((30 * scale, 56 * scale), "holding the numbers" if not in_read else "computing . . .", font=fsmall, fill=MUTED)
        frames.append(img.resize((W, H), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=colors))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], loop=0, duration=int(1000 / fps), optimize=True, disposal=2)
    print(f"[gif] wrote {out} ({os.path.getsize(out)/1e6:.2f} MB)  {disp} {shown}")
    return out


def _load_bundle(path, config):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    model, enc, conn = train_for_viz(config or {})
    return build_bundle(model, enc, conn)


def main():
    ap = argparse.ArgumentParser(description="Render a GIF of the fly solving a problem.")
    ap.add_argument("--bundle", default="viz/fly_bundle.json")
    ap.add_argument("--op", default="+", choices=["+", "-", "*", ">"])
    ap.add_argument("--a", type=int, default=47)
    ap.add_argument("--b", type=int, default=38)
    ap.add_argument("--out", default="viz/fly_solve.gif")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    bundle = _load_bundle(args.bundle, {})
    render_gif(bundle, args.op, args.a, args.b, out=args.out, fps=args.fps)


if __name__ == "__main__":
    main()
