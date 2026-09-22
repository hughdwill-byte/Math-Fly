#!/usr/bin/env python3
"""
End-to-end smoke demo: build a (synthetic) connectome, train the readout on a
few curriculum levels, and print sample predictions. Runs in seconds on CPU
and needs only numpy + scipy. This is the fastest way to confirm the whole
pipeline works before you download the real connectome or scale up.

    python scripts/run_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mathfly.train import train_reservoir
from mathfly.evaluate import evaluate_task
from mathfly.connectome import load_connectome
from mathfly.io_encoding import IOEncoder
from mathfly.model import ReservoirModel
from mathfly.envs import CURRICULUM


def main():
    config = {
        "synthetic_n": 800,          # small = fast; raise for more capacity
        "synthetic_density": 0.03,
        "spectral_radius": 1.15,
        "n_readout": 256,
        "n_train": 300,
        "n_eval": 150,
        "tasks": ["compare", "count", "add_1digit", "sub_1digit", "mul_1digit"],
        "out_dir": "runs/demo",
        "seed": 0,
    }
    print("=== Math-Fly demo: training readout on a synthetic connectome ===\n")
    model, results = train_reservoir(config)

    print("\n=== sample predictions ===")
    for name in ["add_1digit", "sub_1digit"]:
        evaluate_task(model, name, n=100, show=6)


if __name__ == "__main__":
    main()
