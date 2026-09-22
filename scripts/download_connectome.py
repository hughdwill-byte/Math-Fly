#!/usr/bin/env python3
"""
Fetch the Drosophila male-CNS connectome so Math-Fly can use the real wiring.

The male-CNS data lives at https://male-cns.janelia.org/download/ and is also
queryable through neuPrint (https://neuprint.janelia.org, dataset "male-cns").
Because the full release is large and access options change, this script
supports two routes:

  1. --neuprint : query neuPrint's API for a neuron table + connection table and
     write them as CSVs that mathfly.connectome can load directly. Requires a
     free neuPrint auth token (https://neuprint.janelia.org -> account) set in
     the NEUPRINT_TOKEN environment variable, and the `neuprint-python` package
     (`pip install neuprint-python`).

  2. --manual : print exactly which files to download by hand from the portal
     and where to put them. Use this if the API route is unavailable.

Output goes to data/connectome/ as neurons.csv and connections.csv, which
mathfly.connectome.autoload_real() picks up automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

OUT_DIR = "data/connectome"
NEUPRINT_SERVER = "neuprint.janelia.org"
DATASET = "male-cns:v0.9"   # update to the current male-CNS snapshot tag


def via_neuprint(max_neurons: int | None, min_synapses: int):
    try:
        from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria
    except ImportError:
        sys.exit("neuprint-python not installed. Run: pip install neuprint-python")
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        sys.exit("Set NEUPRINT_TOKEN (get one at https://neuprint.janelia.org -> your account).")

    os.makedirs(OUT_DIR, exist_ok=True)
    c = Client(NEUPRINT_SERVER, dataset=DATASET, token=token)
    print(f"[neuprint] connected to {DATASET}")

    # Neuron table: keep traced neurons; grab type + predicted neurotransmitter.
    crit = NeuronCriteria(status="Traced") if max_neurons is None else NeuronCriteria(status="Traced")
    neurons, _ = fetch_neurons(crit, client=c)
    keep_cols = [col for col in ["bodyId", "type", "instance", "predictedNt",
                                 "consensusNt", "cropped", "status"] if col in neurons.columns]
    neurons = neurons[keep_cols]
    if max_neurons is not None:
        neurons = neurons.head(max_neurons)
    neurons.to_csv(os.path.join(OUT_DIR, "neurons.csv"), index=False)
    print(f"[neuprint] wrote {len(neurons)} neurons")

    # Connection table between those neurons.
    body_ids = neurons["bodyId"].tolist()
    _, conns = fetch_adjacencies(sources=body_ids, targets=body_ids, client=c,
                                 min_total_weight=min_synapses)
    conns = conns.rename(columns={"bodyId_pre": "bodyId_pre",
                                  "bodyId_post": "bodyId_post", "weight": "weight"})
    conns.to_csv(os.path.join(OUT_DIR, "connections.csv"), index=False)
    print(f"[neuprint] wrote {len(conns)} connections -> {OUT_DIR}/")
    print("Done. mathfly.connectome.load_connectome() will now use the real data.")


def manual_instructions():
    print(textwrap.dedent(f"""
    ------------------------------------------------------------------
    Manual download of the male-CNS connectome
    ------------------------------------------------------------------
    1. Open  https://male-cns.janelia.org/download/
    2. Download the connectivity / neuron exports (formats vary by
       snapshot; you want two tables):
         * a NEURON table  -- one row per neuron, with a body-id column
           and, ideally, a predicted-neurotransmitter column and a
           cell-type column.
         * a CONNECTION table -- rows of (pre body id, post body id,
           synapse weight).
    3. Save them into:  {os.path.abspath(OUT_DIR)}/
         neurons.csv
         connections.csv
       (Any *neuron*.csv / *connection*.csv names are auto-detected;
        column names are matched flexibly.)
    4. Verify:  python -m mathfly.connectome   # prints stats
       then train:  python -m mathfly.train --mode reservoir

    Tip: start small. In your config set  max_neurons: 5000  and
    keep_types: ["KC","MBON","PN"] to build a mushroom-body learner
    (the fly's associative-learning centre) before scaling up.
    ------------------------------------------------------------------
    """))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neuprint", action="store_true", help="fetch via neuPrint API")
    ap.add_argument("--manual", action="store_true", help="print manual download steps")
    ap.add_argument("--max-neurons", type=int, default=None)
    ap.add_argument("--min-synapses", type=int, default=5)
    args = ap.parse_args()

    if args.neuprint:
        via_neuprint(args.max_neurons, args.min_synapses)
    else:
        manual_instructions()


if __name__ == "__main__":
    main()
