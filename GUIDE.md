# Math-Fly — A Guide to Training the Fly Brain to Do Math

This guide walks you from a cold clone to a connectome-constrained network that
does arithmetic, and then explains how to push its capacity. It's written to be
honest about the science while still letting you do the genuinely cool thing you
asked for.

---

## 0. The one-paragraph reality check (please read)

Janelia's **male-CNS connectome** is a *wiring diagram*: a list of the fly's
neurons and the synapses between them, plus a predicted neurotransmitter per
neuron. It contains **no activity and no learned weights** — it says *who wires
to whom*, not *what the fly thinks*. There is no way to "boot up" a fly's mind
from it, because the thing that would constitute a mind (the pattern of activity
and the tuned synaptic strengths) was never recorded and isn't in the file. So
we do the next best, scientifically real thing: we treat the connectome as a
**fixed architectural prior** — the network may only route information along
wires the fly actually has, respecting excitatory/inhibitory identity (Dale's
law) — and we *train* a small part of the system to solve math. This is the same
idea as Janelia's own connectome-constrained models of the fly visual system
(e.g. [flyvis](https://github.com/TuragaLab/flyvis)). What you get is not a
resurrected fly; it's a biologically-grounded network whose limits you can
actually measure.

**How advanced can the math get?** Empirically, that's set by three knobs, not
by the fly's "IQ":
1. how you **encode** numbers into neural input,
2. how big a **subgraph** of the connectome you use (more neurons = more
   capacity, more RAM/compute), and
3. which **training regime** you use (fixed-reservoir readout vs. full
   backprop-through-time).

Real flies do the bottom of the ladder (comparing "more vs. less" and counting
small sets — this is documented numerosity behavior). Everything above that is
your architecture being pushed past the animal, which is a fair and interesting
experiment as long as you call it that.

---

## 1. Set up the environment

```bash
cd Math-Fly
python -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt                        # numpy, scipy, pyyaml
```

That's all you need for the default (reservoir) training path. Two optional
extras:

```bash
# Backprop-through-time training (learns synaptic gains): needs PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Pulling the REAL connectome from neuPrint
pip install neuprint-python pandas
```

Verify everything works end-to-end on a synthetic stand-in connectome:

```bash
python scripts/run_demo.py
```

You should see `compare` and `count` near 100%, and single-digit arithmetic
well above chance, in a couple of seconds. If that works, the whole pipeline is
healthy and you can move to the real brain.

---

## 2. Get the real fly connectome

You have two routes. Either produces two CSVs in `data/connectome/`:
`neurons.csv` (one row per neuron, with a body-id, a predicted neurotransmitter,
and a cell type) and `connections.csv` (rows of pre-id, post-id, synapse count).
`mathfly` auto-detects these — column names are matched flexibly.

### Route A — neuPrint API (recommended)

1. Make a free account at <https://neuprint.janelia.org> and copy your auth
   token (top-right → *Account*).
2. Export it and download:

   ```bash
   export NEUPRINT_TOKEN="paste-your-token"
   python scripts/download_connectome.py --neuprint --max-neurons 20000
   ```

   `--max-neurons` keeps the highest-degree neurons so you can start small.
   Raise it as your RAM allows (the full CNS is ~140k neurons).

### Route B — manual download

```bash
python scripts/download_connectome.py --manual
```

This prints exactly which files to grab from
<https://male-cns.janelia.org/download/> and where to drop them. Use this if the
API route isn't available to you.

### Sanity-check the loaded brain

```bash
python -m mathfly.connectome        # prints N, density, spectral radius, E/I counts
```

---

## 3. Understand the four moving parts

| Part | File | What it does | Do you train it? |
|------|------|--------------|------------------|
| **Recurrent core** | `connectome.py` | The fly's wiring → signed sparse `W`. Fixed by anatomy. | No (reservoir) / gains only (BPTT) |
| **Sensory/motor I/O** | `io_encoding.py` | Numbers → input pulses on "sensory" neurons; "motor" neuron activity → answer | Frozen input; trained readout |
| **Environment** | `envs.py` | The math curriculum | n/a (it's the task) |
| **Trainer** | `train.py` | Fits the readout (or the gains) per curriculum level | — |

The mental model: **the connectome is the brain, and it stays fixed.** Learning
happens in a thin readout layer — analogous to how the fly's mushroom body has a
large fixed expansion layer (Kenyon cells) feeding a small set of readout
neurons (MBONs) whose weights are what plasticity actually changes during
learning. That biology is why the reservoir approach is a *good* model, not a
shortcut.

---

## 4. Train it — the reservoir way (start here)

This freezes the connectome and trains one linear readout head per task with
ridge regression. It's fast, runs on a CPU, and needs only numpy/scipy.

```bash
# synthetic brain, full curriculum:
python -m mathfly.train --mode reservoir --config configs/quickstart.yaml

# real brain:
python -m mathfly.train --mode reservoir --config configs/full_connectome.yaml
```

What you'll see per curriculum level:

```
[add_1digit  ] train=50.3% test=28.0% (chance 5.3%)  :: a + b, single digit
```

- **train/test** are accuracies; **chance** is the baseline for that level's
  answer range, so you can tell real learning from lucky guessing.
- Each level trains its **own** readout head off the **same** frozen reservoir.
  Heads are saved together in `runs/<name>/readout.npz`.

Inspect predictions:

```bash
python -m mathfly.evaluate runs/quickstart/readout.npz
```

### The curriculum (easiest → hardest)

| Level | Task | Fly-real? |
|-------|------|-----------|
| `compare` | is a > b? | ✅ yes (numerosity comparison) |
| `count` | report a small quantity 0–9 | ✅ roughly |
| `add_1digit` | a + b, single digit | ❌ beyond the animal |
| `sub_1digit` | a − b, non-negative | ❌ |
| `add_2digit` | up to 49+49 | ❌ |
| `mul_1digit` | a × b | ❌ |
| `mixed_easy` | mixed +,−,× single digit | ❌ |
| `add_3digit` | up to 199+199 | ❌ |
| `mixed_hard` | mixed ops up to 20 | ❌ |

Train a subset with `--tasks`:

```bash
python -m mathfly.train --mode reservoir --tasks compare count add_1digit
```

---

## 5. The four training modes

`python -m mathfly.train --mode <mode>` supports four regimes. Start at the top
and move down as you need more power.

| Mode | Command | What learns | Needs | Best for |
|------|---------|-------------|-------|----------|
| `reservoir` | `--mode reservoir --config configs/quickstart.yaml` | one linear readout head per task | numpy | fast sweeps, "what can the raw wiring support?" |
| `digits` | `--mode digits --config configs/digits.yaml` | per-digit readout heads | numpy | **large / multi-digit answers** (see below) |
| `bptt` | `--mode bptt --config configs/bptt.yaml` | per-edge gains + I/O + readout | torch | pushing one hard task past the reservoir ceiling |
| `rl` | `--mode rl --config configs/rl.yaml` | same, from **reward only** | torch + gymnasium | the "agent in a world" framing |

### 5a. Digit-serial (`--mode digits`) — the key to big numbers

A single classifier that has one output per possible answer explodes for large
results (3-digit multiplication has answers up to ~40,000). Digit-serial output
fixes this: the network reads out **each decimal digit independently** with its
own 10-way head, so a 4-digit reader covers 0–9999 with just 4×10 outputs, and
the answer space stops growing with the numbers.

```bash
python -m mathfly.train --mode digits --config configs/digits.yaml
```

It reports **exact-match** and **per-digit** accuracy — watch the units digit
saturate first, then the carries come in as you add capacity. Set `n_digits` to
cover your largest answer.

### 5b. Backprop-through-time (`--mode bptt`)

The reservoir keeps the connectome *entirely* fixed. To also tune the **synaptic
gains** — while keeping every edge's sign and the wiring diagram intact (Dale's
law preserved) — use the PyTorch path. It optimizes a positive per-edge gain +
input weights + readout with cross-entropy, unrolling the dynamics through time.
Slower, one task at a time, but reaches accuracies the fixed reservoir can't. A
GPU is picked up automatically.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m mathfly.train --mode bptt --config configs/bptt.yaml
```

### 5c. Reinforcement learning (`--mode rl`)

The most "living agent" framing: the network is dropped into the
[`MathFlyEnv`](mathfly/gym_env.py) Gymnasium world, sees the problem as timed
sensory input, commits to an answer, and receives **only a scalar reward**
(+1 correct / 0 wrong) — no one tells it the right answer. It's trained with
REINFORCE (policy gradients) + a baseline and an entropy bonus.

```bash
pip install torch gymnasium
python -m mathfly.train --mode rl --config configs/rl.yaml
```

Expect slower, noisier learning than the supervised modes — sparse reward over
many possible answers is intrinsically high-variance. Use RL when the *reward-
driven* framing is the point, or when you extend Math-Fly to a task where you
only have a reward signal. `MathFlyEnv` is a standard `gymnasium.Env` (it passes
`gymnasium`'s `check_env`), so you can also plug in any RL library (SB3, etc.).

---

## 6. How to push the math further (turning the capacity knobs)

If a level is stuck near chance, turn these, roughly in order of payoff:

1. **More neurons.** Raise `max_neurons` (real) or `synthetic_n` (synthetic).
   Capacity scales with the size of the recurrent core. This is the single
   biggest lever — and the honest answer to "how advanced can it get."
2. **Bigger readout population.** Raise `n_readout` so more of the brain's
   activity feeds the answer.
3. **Operating point.** Tune `spectral_radius` (try 1.0–1.5). Too low → the
   network forgets the earlier tokens before it sees `=`; too high → chaotic.
4. **Longer thinking time.** Raise `steps_per_token` and the settle window so
   the recurrence has more integration steps to combine the operands.
5. **Better encoding.** For big numbers, switch from a single-answer classifier
   to **digit-serial decoding** (predict each output digit) — see §7. This is
   how you get past the "answer set is too large to classify" wall.
6. **Regularization.** Tune `ridge` (reservoir) or `lr` (BPTT) if train ≫ test.
7. **Anatomically targeted subgraphs.** Set `keep_types: ["KC","MBON","PN"]` to
   build a **mushroom-body** learner (the fly's actual associative-learning
   center). A well-chosen circuit can beat a bigger random slice.

Rule of thumb for the ceiling: with the fixed reservoir you can expect strong
counting/comparison and meaningful (well-above-chance, not perfect) single-digit
arithmetic; multi-digit and multiplication want either BPTT, a large subgraph,
or digit-serial output — often all three.

---

## 6b. Watch the fly think (the visualiser)

Once you have a trained model, export an interactive page and watch the brain
solve problems you type:

```bash
python -m mathfly.export_web --out viz/fly_viz.html
# open viz/fly_viz.html in any browser (no server needed)
```

What it does: it trains (or you can point it at a config), serialises the
connectome + frozen recurrent weights + sensory embedding + trained readout
heads into the page, and the page **re-runs the exact reservoir dynamics in
JavaScript**. So it's not a canned animation — type a problem, hit *Ask the
fly*, and you watch:

- each digit and the operator arrive as **timed pulses** on the gold sensory
  neurons,
- signal spread through the fly's fixed wiring (teal = excitatory, rose =
  inhibitory), with **sparks travelling along the synapses** (toggle "signal
  trails"); reduced-motion is respected,
- the **readout resolve** during the "thinking" window — decoded **digit by
  digit** on an odometer via the digit-serial heads, so multi-digit answers
  work — with a ✓/✗ against the true value and the head's **real held-out
  accuracy**, so nothing is oversold.

Operand ranges per operator are chosen where a random-reservoir readout is
genuinely competent: comparison (`>`) runs on full two-digit numbers; `+ − ×`
use smaller operands (sequential digit-binding for multi-digit arithmetic is
genuinely hard for a linear readout — see §6a) but still produce multi-digit
answers on the reels.

**Make a GIF.** `python -m mathfly.render_gif --op '*' --a 7 --b 8 --out
viz/fly_solve.gif` renders the same real trajectory to a shareable loop (Pillow),
with neurons, sparks, the token stream, and the answer resolving.

Under the hood, `mathfly/export_web.py`'s `reference_forward()` is a NumPy mirror
of the in-browser math, and a unit test asserts the two agree with the Python
model (for both the digit-serial and compare heads) — so the page can never
silently drift from the real network. To visualise the **real** connectome,
train with `configs/full_connectome.yaml` first (keep `max_neurons` in the low
thousands so the page stays light) and re-export.

## 6c. The real fly, and training its synapses

Everything so far can run on a synthetic stand-in. Here's the real thing.

### Get the real male-CNS connectome

Janelia distributes the male-CNS v1.0 connectome as Arrow (`.feather`) flat
files in a **public Google Cloud bucket** (`gs://flyem-male-cns/`) — no account
or token needed. One command fetches it and builds a compact trainable subgraph:

```bash
pip install pyarrow pandas
python scripts/download_male_cns.py --max-neurons 1200
```

This downloads the neuron annotations, the neurotransmitter predictions (which
set each neuron's excitatory/inhibitory sign, Dale's law), and the ~0.5 GB
body-to-body edge table (25.5M synaptic connections), then extracts the
highest-degree subgraph and caches it to
`data/connectome/male_cns_subgraph.npz` (~0.1 MB, committed with the repo). Any
config with `male_cns: true` then uses the real fly.

### Train just the readout (fast) or the synapses (BPTT)

```bash
# reservoir: fix the real wiring, train the readout heads (seconds)
python -m mathfly.export_web --male-cns --out viz/fly_viz.html

# synapse training: also learn a gain on every real synapse + a per-neuron
# bias, by backprop-through-time (torch; minutes on CPU, faster on GPU)
python -m mathfly.synapse_train --out viz/fly_viz.html
```

`synapse_train` first **warm-starts** the readout from the fast reservoir fit, so
the printed "fixed synapses" line is the real baseline; it then trains the
synapses and keeps the **best-so-far** weights, so it can never ship something
worse than the fixed-wiring baseline.

### What to expect (an honest result)

The fly's real wiring is a **worse arithmetic engine than a same-size random
network** — roughly +16% vs +43% on single-digit addition. That's not a bug: the
connectome is optimised for being a fly, not for symbolic math. What the real fly
*is* good at is **comparing quantities** (~76%), which is a genuinely
fly-plausible behaviour (real flies discriminate "more vs. less"). And on CPU,
short backprop-through-time matches but does not beat the reservoir baseline;
surpassing it wants a GPU and a much longer run. The whole pipeline supports
that — `--male-cns`, larger `--max-neurons`, a GPU (picked up automatically),
more `--rounds` — but this guide won't pretend a CPU demo makes the real fly a
calculator.

## 7. Extending the project

Two of the biggest extensions are now built in — **digit-serial output**
(`--mode digits`, §5a) and a **Gymnasium RL environment + REINFORCE trainer**
(`--mode rl`, §5c). What's left to explore:

- **Curriculum transfer.** Instead of an independent head per level, warm-start
  each level from the previous one (`ReservoirModel.partial_fit_readout`) to
  study transfer between tasks.
- **Read the brain.** Plot which cell **types** the trained readout leans on
  (`connectome.types` indexed by `encoder.readout_neurons`) to see *which fly
  circuits* the math solution recruited — the actually-interesting science.
- **Bigger RL.** Swap REINFORCE for PPO via Stable-Baselines3 on `MathFlyEnv`,
  or vectorize the env for throughput.
- **Spiking dynamics.** Swap `tanh` rates for a leaky integrate-and-fire neuron
  for a more biophysical model.
- **Digit-serial + BPTT.** Combine the two: give the torch RNN `n_digits` heads
  and train with per-digit cross-entropy for multi-digit results end-to-end.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Everything at chance | Lower `spectral_radius` toward 1.0; raise `steps_per_token`; check the connectome loaded (`python -m mathfly.connectome`). |
| Loads synthetic when you wanted real | CSVs not found in `data/connectome/`. Run the download script; confirm filenames contain `neuron`/`connection`. |
| `MemoryError` on the real brain | Lower `max_neurons` or raise `min_synapses` to prune weak edges. |
| BPTT loss is NaN | Lower `lr`; lower `spectral_radius`; shorten sequences. |
| Train high, test low | Raise `ridge`; add more `n_train`; reduce `n_readout`. |

---

## 9. Where things are

- Configs: `configs/*.yaml` — `quickstart`, `full_connectome`, `digits`, `bptt`,
  `rl`. Copy and edit; every knob above is exposed.
- Modes: `python -m mathfly.train --mode {reservoir|digits|bptt|rl}`.
- Results: `runs/<name>/results.json` (metrics) and `readout.npz` / `*.pt`.
- RL world: `mathfly/gym_env.py` (`MathFlyEnv`, a standard `gymnasium.Env`).
- Tests: `python -m pytest tests/` — 8 tests; confirm the pipeline learns above
  chance, digit round-trips, and the env is Gymnasium-compliant.

Happy training. 🪰➕
