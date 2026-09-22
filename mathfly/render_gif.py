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
