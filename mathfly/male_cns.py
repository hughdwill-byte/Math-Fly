"""
Build a trainable network from the REAL Drosophila male-CNS connectome.

Data source: Janelia FlyEM male-CNS v1.0, distributed as Apache Arrow (.feather)
flat files in the public Google Cloud Storage bucket gs://flyem-male-cns/. See
scripts/download_male_cns.py to fetch them. We use three files:

  * connectome-weights ... traced-only.feather -- body-to-body edges
      columns: body_pre, body_post, weight (synapse count), type_pre, type_post
  * body-neurotransmitters ....feather         -- per-neuron predicted NT
      columns: body, consensus_nt / predicted_nt   (-> excitatory/inhibitory sign)
  * body-annotations ....feather               -- per-neuron cell type / class

The full graph (25.5M edges, 200k+ neurons) is far too large to train a
recurrent network on directly, so we extract a subgraph of the highest-degree
neurons (the connectivity hubs of the central brain), apply Dale's law from the
predicted neurotransmitters, and cache it as a compact .npz that
mathfly.connectome can load like any other Connectome.

This is a genuine slice of the real fly's wiring -- not synthetic.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import sparse

from .connectome import Connectome

# Glutamate is predominantly inhibitory in the fly CNS; GABA inhibitory;
# acetylcholine (the majority) excitatory; modulators treated as excitatory.
INHIBITORY_NT = {"gaba", "glutamate", "glut"}


def _nt_sign(nt: str) -> int:
    return -1 if str(nt).strip().lower() in INHIBITORY_NT else 1


def build_subgraph(raw_dir: str = "data/male_cns_raw",
                   max_neurons: int = 1200,
                   min_synapses: int = 5,
                   include_superclasses: list[str] | None = None,
                   cache: str | None = "data/connectome/male_cns_subgraph.npz",
                   verbose: bool = True) -> Connectome:
    """Extract a signed subgraph of the top-degree neurons from the male-CNS
    feather files and (optionally) cache it to `cache`.

    include_superclasses: neuron superclasses to force-include regardless of
        degree, e.g. ["vnc_motor", "descending_neuron"] to keep the fly's real
        motor/command system (its 'basic movement' hardware) in the subgraph."""
    import pyarrow.feather as feather

    wpath = _find(raw_dir, ["weights", "connectome-weights"])
    ntpath = _find(raw_dir, ["neurotransmitter"])
    anpath = _find(raw_dir, ["annotation"])
    if wpath is None:
        raise FileNotFoundError(
            f"No connectome-weights feather in {raw_dir}. Run "
            "scripts/download_male_cns.py first.")

    # optionally force-include whole superclasses (e.g. the motor system)
    forced = set()
    forced_super = {}
    if include_superclasses and anpath:
        a0 = feather.read_table(anpath, columns=["bodyId", "superclass"],
                                memory_map=True).to_pandas()
        sel = a0[a0["superclass"].isin(include_superclasses)]
        forced = set(sel["bodyId"].tolist())
        forced_super = dict(zip(sel["bodyId"].to_numpy(), sel["superclass"].to_numpy()))
        if verbose:
            print(f"[male-cns] force-including {len(forced)} neurons from "
                  f"{include_superclasses}")

    if verbose:
        print(f"[male-cns] reading edges from {os.path.basename(wpath)} ...")
    tbl = feather.read_table(wpath, columns=["body_pre", "body_post", "weight"],
                             memory_map=True)
    pre = tbl.column("body_pre").to_numpy()
    post = tbl.column("body_post").to_numpy()
    w = tbl.column("weight").to_numpy()

    # denoise weak connections
    keep = w >= min_synapses
    pre, post, w = pre[keep], post[keep], w[keep]
    if verbose:
        print(f"[male-cns] {len(w):,} edges after min_synapses>={min_synapses}")

    # degree per body (in + out synapse count) -> pick the hubs
    bodies, inv = np.unique(np.concatenate([pre, post]), return_inverse=True)
    deg = np.zeros(len(bodies), dtype=np.int64)
    half = len(pre)
    np.add.at(deg, inv[:half], w)
    np.add.at(deg, inv[half:], w)
    order = bodies[np.argsort(deg)[::-1]]
    forced_present = [b for b in order if b in forced]     # keep motor system
    top_set = set(forced_present)
    for b in order:
        if len(top_set) >= max_neurons:
            break
        top_set.add(int(b))
    if verbose:
        print(f"[male-cns] selected {len(top_set):,} neurons by degree"
              + (f" (incl. {len(forced_present)} motor/command)" if forced else ""))

    # filter edges to the subgraph
    m = np.fromiter((p in top_set and q in top_set for p, q in zip(pre, post)),
                    count=len(pre), dtype=bool)
    pre, post, w = pre[m], post[m], w[m]

    body_ids = np.array(sorted(top_set))
    index = {b: i for i, b in enumerate(body_ids)}
    n = len(body_ids)

    # neurotransmitter -> sign per neuron (Dale's law)
    sign = np.ones(n, dtype=np.float32)
    types = np.array(["unknown"] * n, dtype=object)
    if ntpath:
        nt = feather.read_table(ntpath, memory_map=True).to_pandas()
        col = "consensus_nt" if "consensus_nt" in nt.columns else "predicted_nt"
        nt = nt[["body", col]].dropna()
        sub = nt[nt["body"].isin(top_set)]
        for b, v in zip(sub["body"].to_numpy(), sub[col].to_numpy()):
            if b in index:
                sign[index[b]] = _nt_sign(v)
    if anpath:
        an = feather.read_table(anpath, columns=["bodyId", "type"], memory_map=True).to_pandas()
        sub = an[an["bodyId"].isin(top_set)]
        for b, t in zip(sub["bodyId"].to_numpy(), sub["type"].to_numpy()):
            if b in index and isinstance(t, str):
                types[index[b]] = t

    # signed sparse matrix W[post, pre] = sign(pre) * log1p(synapses)
    rows = np.array([index[q] for q in post])
    cols = np.array([index[p] for p in pre])
    data = (sign[cols] * np.log1p(w.astype(np.float32))).astype(np.float32)
    W = sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates(); W = W.tocsr()

    # movement neurons: real motor/command neurons kept in the subgraph
    movement_idx = np.array([index[b] for b in forced if b in index], dtype=np.int64)
    conn = Connectome(W=W, sign=sign, body_ids=body_ids, types=types, source="real",
                      meta={"dataset": "male-cns-v1.0", "n_neurons": n,
                            "n_synapses": int(W.nnz), "max_neurons": max_neurons,
                            "min_synapses": min_synapses,
                            "movement_idx": movement_idx.tolist()})
    n_inh = int((sign < 0).sum())
    if verbose:
        print(f"[male-cns] subgraph: {n} neurons, {W.nnz:,} edges, "
              f"{n_inh} inhibitory ({n_inh/n:.0%}), source=REAL fly")
    if cache:
        save_subgraph(conn, cache)
        if verbose:
            print(f"[male-cns] cached -> {cache}")
    return conn


def save_subgraph(conn: Connectome, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    W = conn.W.tocoo()
    np.savez_compressed(path, row=W.row, col=W.col, data=W.data, n=conn.n,
                        sign=conn.sign, body_ids=conn.body_ids,
                        types=conn.types.astype(str),
                        movement_idx=np.array(conn.meta.get("movement_idx", []), dtype=np.int64))


def load_subgraph(path: str) -> Connectome:
    d = np.load(path, allow_pickle=True)
    n = int(d["n"])
    W = sparse.coo_matrix((d["data"], (d["row"], d["col"])), shape=(n, n),
                          dtype=np.float32).tocsr()
    mv = d["movement_idx"].tolist() if "movement_idx" in d.files else []
    return Connectome(W=W, sign=d["sign"].astype(np.float32),
                      body_ids=d["body_ids"], types=d["types"].astype(object),
                      source="real",
                      meta={"dataset": "male-cns-v1.0", "n_neurons": n,
                            "n_synapses": int(W.nnz), "movement_idx": mv})


def _find(d, keys):
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        low = f.lower()
        if f.endswith(".feather") and any(k in low for k in keys):
            return os.path.join(d, f)
    return None


if __name__ == "__main__":
    c = build_subgraph()
    c.rescale_spectral_radius(1.1)
    print("spectral radius ~", round(c.spectral_radius(), 3))
