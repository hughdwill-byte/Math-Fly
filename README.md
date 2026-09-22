# 🪰 Math-Fly

Train a **Drosophila (fruit fly) connectome-constrained neural network** to do
math. The network's recurrent wiring is taken from Janelia's
[male-CNS connectome](https://male-cns.janelia.org/download/) — a map of which
of the fly's ~140,000 neurons wire to which — and dropped into a virtual
environment of arithmetic tasks that get progressively harder. You then train a
small readout (or, optionally, the synaptic gains themselves) to turn the fly's
circuitry into a calculator, and watch how far its capacity goes.

> **Read this first — what this is and isn't.** The connectome is a *wiring
> diagram*, not a living, thinking brain. It contains no recorded activity and
> no learned weights: it tells you the fly's neurons and synapses, not what the
> fly computes. Math-Fly uses that wiring as a fixed architectural prior for a
> trainable network. So this is **not** "reviving a fruit fly and teaching it
> algebra." It's a scientifically legitimate thing in the same spirit as
> Janelia's own [flyvis](https://github.com/TuragaLab/flyvis) work: build a
> network the fly's real anatomy constrains, and see what it can learn.
> Real flies genuinely do the *easy* end of the curriculum (comparing and
> counting small quantities); the symbolic multi-digit levels are a stress test
> of the architecture, not a claim about the animal. See
> [`GUIDE.md`](GUIDE.md) for the full picture.

## Quickstart (runs in seconds, no download needed)

```bash
pip install -r requirements.txt          # numpy + scipy + pyyaml is enough
python scripts/run_demo.py               # build synthetic connectome, train, evaluate
```

Expected output (a synthetic stand-in connectome, so you can develop offline):

```
[connectome] source=synthetic N=800 synapses=18895 rho~=1.15
[compare     ] train=99.3% test=98.0% (chance 33.3%)
[count       ] train=100.0% test=100.0% (chance 9.1%)
[add_1digit  ] train=50.3% test=28.0% (chance 5.3%)
[sub_1digit  ] train=62.7% test=44.7% (chance 10.0%)
[mul_1digit  ] train=76.3% test=53.3% (chance 1.2%)
```

## Watch it think 🧠

Export a **standalone, interactive web page** where you type a problem and watch
the fly's neurons fire as it solves it — teal for excitatory, rose for
inhibitory, gold for the sensory neurons the digits arrive on. The page embeds
the real trained weights and re-runs the exact reservoir dynamics **in the
browser**, so the answer you see emerge is the model's genuine output (it even
shows a ✓/✗ — a small brain sometimes guesses wrong, which is real).

```bash
python -m mathfly.export_web --out viz/fly_viz.html   # trains + builds the page
# then just open viz/fly_viz.html in any browser
```

*How to read it:* digits pulse into the gold sensory neurons (top-right stream),
signal spreads through the fixed fly wiring, and during the "thinking" window the
readout neurons' activity is decoded into the answer — which you watch resolve in
the side panel.

Then swap in the **real** fly connectome:

```bash
python scripts/download_connectome.py --manual        # how to get the data
# or, with a free neuPrint token:
python scripts/download_connectome.py --neuprint --max-neurons 20000
python -m mathfly.train --mode reservoir --config configs/full_connectome.yaml
```

## How it works

```
 math problem            fixed connectome                 trained
 "3 + 4 ="   ─encode─▶   recurrent network   ─readout─▶   answer "7"
 (token pulses)          (the fly's wiring;              (a small linear
                          who-wires-to-whom is            head we learn —
                          frozen, Dale's law kept)        the "brain" stays fixed)
```

1. **Connectome → recurrent core** (`mathfly/connectome.py`). Each neuron
   becomes a unit; each synapse becomes an edge whose **sign** comes from the
   presynaptic neuron's predicted neurotransmitter (Dale's law) and whose
   **magnitude** is the synapse count. The result is a signed sparse matrix
   `W`, rescaled to a chosen spectral radius so the dynamics are rich but
   stable.
2. **Virtual environment** (`mathfly/envs.py`). A curriculum of tasks from
   fly-plausible (compare/count small numbers) up to multi-digit mixed
   arithmetic.
3. **I/O** (`mathfly/io_encoding.py`). Numbers enter as timed input pulses on a
   set of "sensory" neurons (place code + magnitude code); the answer is read
   from a set of "motor" neurons.
4. **Training** (`mathfly/train.py`). Four regimes via `--mode`:
   - **`reservoir`** (default, NumPy, CPU): freeze the connectome, train one
     linear readout head per task via ridge regression. Fast and always works.
   - **`digits`** (NumPy): decode the answer **digit-by-digit** (one 10-way head
     per position) so answer size no longer caps the math — the key to big /
     multi-digit results.
   - **`bptt`** (PyTorch): keep the connectome's sign + sparsity fixed but learn
     a per-edge gain + I/O by backprop-through-time. Stronger, needs compute.
   - **`rl`** (PyTorch + Gymnasium): drop the network into `MathFlyEnv` and train
     it from **reward only** with REINFORCE — the "agent in a world" framing.

## Layout

```
mathfly/
  connectome.py    load male-CNS CSVs (or synthetic fallback) -> signed sparse W
  io_encoding.py   encode numbers as neural input; single + digit-serial decode
  model.py         ReservoirModel (numpy) + build_torch_rnn (BPTT/RL policy)
  envs.py          the math curriculum (compare -> multi-digit mixed)
  gym_env.py       MathFlyEnv: a Gymnasium RL world around the curriculum
  train.py         training loop + CLI (reservoir | digits | bptt | rl)
  rl_train.py      REINFORCE trainer (network as reward-driven agent)
  evaluate.py      accuracy + sample predictions
  export_web.py    export a trained model to the interactive visualiser
scripts/
  download_connectome.py   fetch real data (neuPrint API or manual steps)
  run_demo.py              end-to-end smoke demo
viz/
  fly_viz_template.html    the interactive brain player (weights injected in)
  fly_viz.html             generated standalone visualiser (open in a browser)
configs/           quickstart, full_connectome, digits, bptt, rl (.yaml)
tests/             pytest smoke tests (9, all green)
GUIDE.md           ← full training guide (start here)
```

## Requirements

- Python ≥ 3.10, `numpy`, `scipy`, `pyyaml` (core).
- Optional `torch` for BPTT: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Optional `neuprint-python` + `pandas` to pull the real connectome.

See [`GUIDE.md`](GUIDE.md) for the step-by-step training guide, how far the math
can realistically go, and how to push the capacity.
