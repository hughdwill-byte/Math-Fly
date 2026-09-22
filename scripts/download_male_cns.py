#!/usr/bin/env python3
"""
Download the REAL Drosophila male-CNS connectome (Janelia FlyEM v1.0) and build
a trainable subgraph for Math-Fly.

The connectome is distributed as Apache Arrow (.feather) flat files in the
PUBLIC Google Cloud Storage bucket gs://flyem-male-cns/ -- no account or token
needed, unlike the neuPrint API. This script pulls the three files we need over
plain HTTPS and builds the cached subgraph.

    python scripts/download_male_cns.py                 # default 1200-neuron subgraph
    python scripts/download_male_cns.py --max-neurons 2500

Files fetched (into data/male_cns_raw/):
    body-annotations ...        (~14 MB)  cell types / classes
    body-neurotransmitters ...  (~43 MB)  predicted NT -> excitatory/inhibitory
    connectome-weights traced   (~500 MB) body-to-body synaptic edges

Then a compact signed subgraph is cached to
data/connectome/male_cns_subgraph.npz, which mathfly loads with
`load_connectome({"male_cns": True})` or any config with `male_cns: true`.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
RAW = "data/male_cns_raw"
FILES = {
    "annotations.feather": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters.feather": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights.feather": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
}


def _download(url, dest):
    tmp = dest + ".part"
    print(f"  fetching {os.path.basename(dest)} ...", flush=True)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); got += len(chunk)
            if total:
                print(f"\r    {got/1e6:7.1f} / {total/1e6:.1f} MB", end="", flush=True)
        print()
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-neurons", type=int, default=1200,
                    help="size of the subgraph to build (top-degree neurons)")
    ap.add_argument("--min-synapses", type=int, default=5)
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep the large raw feather files after building the subgraph")
    args = ap.parse_args()

    try:
        import pyarrow  # noqa
    except ImportError:
        sys.exit("Please `pip install pyarrow pandas` first (the .feather reader).")

    os.makedirs(RAW, exist_ok=True)
    for local, remote in FILES.items():
        dest = os.path.join(RAW, local)
        if os.path.exists(dest):
            print(f"  {local} already present, skipping")
            continue
        _download(f"{BUCKET}/{remote}", dest)

    print("Building subgraph ...")
    from mathfly.male_cns import build_subgraph
    build_subgraph(raw_dir=RAW, max_neurons=args.max_neurons,
                   min_synapses=args.min_synapses)

    if not args.keep_raw:
        w = os.path.join(RAW, "weights.feather")
        if os.path.exists(w):
            os.remove(w)          # the 500 MB edge file isn't needed once cached
            print(f"  removed {w} (subgraph cached; pass --keep-raw to keep it)")
    print("\nDone. Train the real fly with:")
    print("  python -m mathfly.synapse_train --out viz/fly_viz.html   # trains the synapses")
    print("  # or the fast reservoir path:")
    print("  python -m mathfly.export_web --male-cns --out viz/fly_viz.html")


if __name__ == "__main__":
    # allow running as a script from repo root
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
