"""
A basic-movement skill for the fly, read out from its REAL motor neurons.

The user's brief: the fly can forget its other functions as long as it still
"remembers basic movement". The male-CNS subgraph keeps the fly's actual motor
and descending (command) neurons (superclasses vnc_motor / descending_neuron;
see mathfly.male_cns build_subgraph(include_superclasses=...)). This module
trains a small readout on those motor neurons that maps a direction command
(forward / left / right / stop) to the right motor output -- so the network can
still drive movement even after the rest of its synapses are retrained for math.

It's a deliberately simple, honest capability: there is no body or physics here,
so "movement" means the fly reliably activates its motor neurons in a
command-specific, decodable way. The visualiser lets you press a direction and
watch the real motor neurons fire.
"""

from __future__ import annotations

import numpy as np

COMMANDS = ["forward", "left", "right", "stop"]


def _command_embed(n_input: int, seed: int = 7) -> np.ndarray:
    """A fixed sparse input pattern per movement command (like the token embed)."""
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 1, size=(len(COMMANDS), n_input)).astype(np.float32)
    E *= (rng.random((len(COMMANDS), n_input)) < 0.5)
    return E


def _encode_command(enc, cmd_embed, cmd_id, N):
    """Build the (T, N) input drive for a command: hold it, then settle."""
    spt = enc.steps_per_token
    hold = 2 * spt
    T = hold + enc.settle_tokens * spt
    U = np.zeros((T, N), dtype=np.float32)
    vec = cmd_embed[cmd_id] * enc.input_gain
    for t in range(hold):
        U[t, enc.input_neurons] = vec
    # settle: let it ring down (no drive) while we read the motor neurons
    return U, T


def train_movement(model, enc, motor_idx, cmd_embed=None, n_train=300, ridge=1e-2,
                   seed=7):
    """Train a ridge readout mapping command -> direction, read from the motor
    neurons. Returns (W_move, cmd_embed, motor_idx, accuracy)."""
    if cmd_embed is None:
        cmd_embed = _command_embed(enc.n_input, seed)
    motor_idx = np.asarray(motor_idx, dtype=np.int64)
    if motor_idx.size == 0:                       # no motor neurons in this graph
        return None

    def feats(cmd_id):
        U, T = _encode_command(enc, cmd_embed, cmd_id, model.C.n)
        R = model.run(U)
        w = enc.settle_tokens * enc.steps_per_token
        f = R[T - w:][:, motor_idx].mean(axis=0)
        return np.concatenate([f, [1.0]]).astype(np.float32)

    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_train):
        c = int(rng.integers(0, len(COMMANDS)))
        X.append(feats(c)); Y.append(c)
    X = np.stack(X); Yv = np.array(Y)
    Yoh = np.zeros((len(Yv), len(COMMANDS)), dtype=np.float32); Yoh[np.arange(len(Yv)), Yv] = 1
    F = X.shape[1]
    W_move = np.linalg.solve(X.T @ X + ridge * np.eye(F, dtype=np.float32), X.T @ Yoh).T
    # accuracy on fresh commands
    te = np.random.default_rng(seed + 1); correct = 0; n = 200
    for _ in range(n):
        c = int(te.integers(0, len(COMMANDS)))
        correct += int(np.argmax(W_move @ feats(c)) == c)
    acc = correct / n
    return {"W_move": W_move, "cmd_embed": cmd_embed, "motor_idx": motor_idx, "accuracy": acc}


def add_to_bundle(bundle: dict, mv: dict, round_to: int = 4):
    """Attach a trained movement skill to a visualiser bundle."""
    if mv is None:
        return bundle

    def r(a):
        return np.round(np.asarray(a, dtype=np.float64), round_to).tolist()

    bundle["movement"] = {
        "commands": COMMANDS,
        "embed": r(mv["cmd_embed"]),               # (n_cmd, n_input)
        "motor_neurons": mv["motor_idx"].astype(int).tolist(),
        "W_move": r(mv["W_move"]),                 # (n_cmd, n_motor+1)
        "accuracy": round(float(mv["accuracy"]), 3),
    }
    return bundle
