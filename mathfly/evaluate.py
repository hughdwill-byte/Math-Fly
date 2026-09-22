"""Evaluate a trained reservoir model and show sample predictions.

    python -m mathfly.evaluate runs/quickstart        # a run directory
    python -m mathfly.evaluate runs/quickstart/readout.npz

The evaluator rebuilds the *exact* connectome + encoder the run used (read from
the run's results.json), so the saved readout heads line up with the features.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .envs import CURRICULUM, CURRICULUM_BY_NAME, make_dataset


def evaluate_task(model, task_name: str, n: int = 200, seed: int = 12345, show: int = 8):
    task = CURRICULUM_BY_NAME[task_name]
    probs, ans = make_dataset(task, n, seed=seed)
    # Use the readout head trained for this task if one exists.
    head = task_name if task_name in getattr(model, "readouts", {}) else None
    preds = [model.predict(p, name=head) for p in probs]
    acc = float(np.mean(np.array(preds) == ans))
    print(f"\n== {task_name} == accuracy {acc:.1%} on {n} problems ==")
    for p, a, pr in list(zip(probs, ans, preds))[:show]:
        mark = "OK " if a == pr else "XX "
        print(f"  {mark}{''.join(p):10s} -> pred {pr:4d}  (true {a})")
    return acc


def rebuild_model(run_path: str):
    """Reconstruct a trained ReservoirModel from a run directory (or the
    readout.npz inside one), using the config saved in results.json."""
    from .connectome import load_connectome
    from .io_encoding import IOEncoder
    from .model import ReservoirModel

    run_dir = run_path if os.path.isdir(run_path) else os.path.dirname(run_path)
    npz = run_path if run_path.endswith(".npz") else os.path.join(run_dir, "readout.npz")
    with open(os.path.join(run_dir, "results.json")) as f:
        config = json.load(f)["config"]

    conn = load_connectome(config)
    # n_answers must match what training used (max over the trained tasks).
    trained = config.get("tasks") or [t.name for t in CURRICULUM]
    max_answer = max(CURRICULUM_BY_NAME[t].max_answer for t in trained) + 1
    enc = IOEncoder(
        n_neurons=conn.n,
        n_input=config.get("n_input", 64),
        n_readout=config.get("n_readout", 256),
        n_answers=max_answer,
        steps_per_token=config.get("steps_per_token", 4),
        input_gain=config.get("input_gain", 1.0),
        seed=config.get("seed", 0),
    )
    model = ReservoirModel(conn, enc, alpha=config.get("alpha", 0.3),
                           noise=config.get("noise", 0.0), seed=config.get("seed", 0))
    model.load(npz)
    return model, config


if __name__ == "__main__":
    import sys
    run_path = sys.argv[1] if len(sys.argv) > 1 else "runs/quickstart"
    model, config = rebuild_model(run_path)
    for name in list(model.readouts) or ["add_1digit", "sub_1digit", "mul_1digit"]:
        evaluate_task(model, name, show=6)
