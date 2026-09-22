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

## 5. Train it harder — backprop-through-time (optional)

The reservoir keeps the connectome *entirely* fixed. To let learning also tune
the **synaptic gains** — while still keeping every edge's sign and the wiring
diagram intact (Dale's law preserved) — use the PyTorch path:

```bash
python -m mathfly.train --mode bptt --config configs/bptt.yaml
```

This optimizes a positive per-edge gain + the input weights + the readout with
cross-entropy, unrolling the recurrent dynamics through time. It's slower and
one-task-at-a-time, but it reaches accuracies the fixed reservoir can't. Use a
GPU if you have one (it's picked up automatically).

**When to use which:**
- *Reservoir* — fast experiments, whole-curriculum sweeps, "how much can the raw
  wiring support?", limited compute.
- *BPTT* — "what's the ceiling if the fly's circuit could also adapt its synaptic
  strengths?", single hard task, you have a GPU.

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

## 7. Extending the project

- **Digit-serial output.** Replace the single softmax with one readout head per
  output digit position (`io_encoding.py` → `target_onehot`, and give the model
  several heads). This makes the answer space size-independent and unlocks large
  numbers. This is the highest-value extension.
- **True RL environment.** The tasks are supervised now. Wrap them as a
  `gymnasium` env (reward for correct motor output) and train with policy
  gradients for a more "agent living in a world" framing. `pip install .[rl]`.
- **Curriculum transfer.** Instead of an independent head per level, warm-start
  each level from the previous one (`partial_fit_readout`) to study transfer.
- **Read the brain.** Plot which cell **types** the trained readout leans on
  (`connectome.types` indexed by `encoder.readout_neurons`) to see *which fly
  circuits* the math solution recruited — the actually-interesting science.
- **Spiking dynamics.** Swap `tanh` rates for a leaky integrate-and-fire neuron
  for a more biophysical model.

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

- Configs: `configs/*.yaml` — copy and edit these; every knob above is exposed.
- Results: `runs/<name>/results.json` (metrics) and `readout.npz` (trained heads).
- Tests: `python -m pytest tests/` — confirms the pipeline learns above chance.

Happy training. 🪰➕
