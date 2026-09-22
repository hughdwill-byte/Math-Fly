"""
Load the Drosophila male-CNS connectome into a signed, sparse recurrent
weight matrix that can be used as the recurrent core of a neural network.

The Janelia male-CNS release (https://male-cns.janelia.org/download/) is a
*wiring diagram*: a list of neurons and a list of synaptic connections
between them, plus a predicted neurotransmitter for each neuron. It does NOT
contain any recorded activity or learned weights -- it tells you *who wires to
whom*, not *what the fly computes*. We turn that wiring diagram into an
architectural prior for a trainable network:

    * nodes      -> model neurons
    * a synapse  -> an edge whose SIGN comes from the presynaptic neuron's
                    predicted neurotransmitter (Dale's law: a neuron is either
                    excitatory or inhibitory for all of its outputs) and whose
                    MAGNITUDE is proportional to the synapse count.

Two data sources are supported:

  1. Real connectome CSVs downloaded from the male-CNS / neuPrint portal
     (see scripts/download_connectome.py). Column names are auto-detected and
     flexible, because export formats vary between snapshots.

  2. A synthetic, biologically-plausible fallback graph (Dale's law, ~80/20
     E/I split, sparse log-normal weights). This lets the entire pipeline run
     with zero downloads so you can develop and smoke-test immediately, then
     swap in the real thing.

Everything downstream (model.py, train.py) only sees the resulting
`Connectome` object, so real vs. synthetic is transparent.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import sparse

# Neurotransmitter -> sign. Everything not listed defaults to excitatory,
# which is the majority class and a safe prior for unpredicted neurons.
INHIBITORY_NT = {"gaba", "glutamate", "glut"}  # glutamate is mostly inhibitory in the fly CNS
EXCITATORY_NT = {"acetylcholine", "ach", "dopamine", "octopamine", "serotonin", "5ht"}


@dataclass
class Connectome:
    """A signed sparse recurrent connectivity matrix plus node metadata."""

    W: sparse.csr_matrix          # (N, N) signed weights; W[i, j] = pre j -> post i
    sign: np.ndarray              # (N,) +1 excitatory / -1 inhibitory per neuron
    body_ids: np.ndarray          # (N,) original connectome ids (or synthetic indices)
    types: np.ndarray             # (N,) cell-type label strings (may be "unknown")
    source: str = "synthetic"     # "real" or "synthetic"
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.W.shape[0]

    @property
    def density(self) -> float:
        return self.W.nnz / (self.n * self.n)

    def spectral_radius(self, k: int = 6) -> float:
        """Largest-magnitude eigenvalue estimate; governs dynamical stability."""
        if self.n <= k + 2:
            vals = np.linalg.eigvals(self.W.toarray())
            return float(np.max(np.abs(vals)))
        try:
            from scipy.sparse.linalg import eigs
            vals = eigs(self.W.astype(np.float64), k=k, which="LM",
                        return_eigenvectors=False, maxiter=2000)
            return float(np.max(np.abs(vals)))
        except Exception:
            # Fallback: cheap power-iteration bound.
            v = np.random.randn(self.n)
            for _ in range(50):
                v = self.W @ v
                nrm = np.linalg.norm(v)
                if nrm == 0:
                    return 0.0
                v /= nrm
            return float(np.linalg.norm(self.W @ v))

    def rescale_spectral_radius(self, target: float) -> "Connectome":
        """Rescale recurrent weights so the network sits at a chosen edge of
        chaos. target < 1 => stable/contracting; ~1.0..1.5 => rich dynamics."""
        rho = self.spectral_radius()
        if rho > 0:
            self.W = self.W * (target / rho)
            self.meta["spectral_radius"] = target
        return self


# --------------------------------------------------------------------------- #
# Real connectome loading
# --------------------------------------------------------------------------- #

def _find_column(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    # substring match
    for c in columns:
        cl = c.lower()
        if any(cand in cl for cand in candidates):
            return c
    return None


def load_real_connectome(
    neurons_csv: str,
    connections_csv: str,
    min_synapses: int = 5,
    max_neurons: Optional[int] = None,
    keep_types: Optional[list[str]] = None,
) -> Connectome:
    """Build a Connectome from male-CNS / neuPrint CSV exports.

    Parameters
    ----------
    neurons_csv : path to a table with (at minimum) a body-id column and,
        ideally, a predicted-neurotransmitter column and a type column.
    connections_csv : path to a table of (pre body id, post body id, weight),
        where weight is the synapse count.
    min_synapses : drop weak connections below this synapse count (denoising).
    max_neurons : keep only the N highest-degree neurons (subsample the graph
        to fit your compute budget). None = keep all.
    keep_types : optional list of cell-type substrings to restrict to (e.g.
        ["KC", "MBON", "PN"] to build a mushroom-body-centric learner).
    """
    import pandas as pd

    neurons = pd.read_csv(neurons_csv)
    conns = pd.read_csv(connections_csv)

    nid = _find_column(neurons.columns, ["bodyid", "body_id", "id", "root_id", "pre_root_id"])
    nnt = _find_column(neurons.columns, ["predictednt", "nt", "ntype", "neurotransmitter", "consensusnt"])
    ntype = _find_column(neurons.columns, ["type", "celltype", "cell_type", "instance"])
    if nid is None:
        raise ValueError(f"Could not find a body-id column in {neurons_csv}: {list(neurons.columns)}")

    pre = _find_column(conns.columns, ["bodyid_pre", "pre", "pre_id", "source", "bodyidpre", "pre_root_id"])
    post = _find_column(conns.columns, ["bodyid_post", "post", "post_id", "target", "bodyidpost", "post_root_id"])
    wcol = _find_column(conns.columns, ["weight", "synapses", "count", "syn_count", "n_syn"])
    if pre is None or post is None:
        raise ValueError(f"Could not find pre/post columns in {connections_csv}: {list(conns.columns)}")
    if wcol is None:
        conns["_w"] = 1
        wcol = "_w"

    # Optional cell-type filter.
    if keep_types and ntype is not None:
        mask = neurons[ntype].astype(str).str.contains("|".join(keep_types), case=False, na=False)
        neurons = neurons[mask]

    # Sign per neuron from predicted neurotransmitter.
    def nt_to_sign(v):
        s = str(v).strip().lower()
        if s in INHIBITORY_NT:
            return -1
        return 1  # excitatory default (acetylcholine-dominant CNS)

    sign_map = {}
    type_map = {}
    for _, row in neurons.iterrows():
        bid = row[nid]
        sign_map[bid] = nt_to_sign(row[nnt]) if nnt is not None else 1
        type_map[bid] = str(row[ntype]) if ntype is not None else "unknown"

    # Denoise + restrict to neurons we kept.
    conns = conns[conns[wcol] >= min_synapses]
    valid = set(sign_map.keys())
    conns = conns[conns[pre].isin(valid) & conns[post].isin(valid)]

    # Optionally subsample to the highest-degree neurons.
    if max_neurons is not None:
        deg = {}
        for b, w in zip(conns[pre], conns[wcol]):
            deg[b] = deg.get(b, 0) + w
        for b, w in zip(conns[post], conns[wcol]):
            deg[b] = deg.get(b, 0) + w
        top = sorted(deg, key=deg.get, reverse=True)[:max_neurons]
        keep = set(top)
        conns = conns[conns[pre].isin(keep) & conns[post].isin(keep)]
        body_ids = np.array(sorted(keep))
    else:
        used = set(conns[pre]) | set(conns[post])
        body_ids = np.array(sorted(used))

    index = {b: i for i, b in enumerate(body_ids)}
    n = len(body_ids)
    if n == 0:
        raise ValueError("No neurons survived filtering; relax min_synapses / max_neurons.")

    sign = np.array([sign_map[b] for b in body_ids], dtype=np.float32)
    types = np.array([type_map[b] for b in body_ids], dtype=object)

    # Build signed sparse matrix W[post, pre] = sign(pre) * synapse_count.
    rows, cols, data = [], [], []
    for p, q, w in zip(conns[pre], conns[post], conns[wcol]):
        i, j = index[q], index[p]         # post=row, pre=col
        rows.append(i); cols.append(j)
        data.append(sign_map[p] * float(w))
    W = sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()

    # Normalise magnitudes: log-compress synapse counts so a few giant
    # connections don't dominate, then scale to unit-ish per-neuron input.
    W.data = np.sign(W.data) * np.log1p(np.abs(W.data))

    return Connectome(
        W=W, sign=sign, body_ids=body_ids, types=types, source="real",
        meta={"n_neurons": n, "n_synapses": int(W.nnz),
              "min_synapses": min_synapses, "keep_types": keep_types},
    )


def autoload_real(data_dir: str, **kwargs) -> Optional[Connectome]:
    """Try to find neuron/connection CSVs in a directory and load them."""
    neurons = (glob.glob(os.path.join(data_dir, "*neuron*.csv"))
               or glob.glob(os.path.join(data_dir, "*nodes*.csv")))
    conns = (glob.glob(os.path.join(data_dir, "*connection*.csv"))
             or glob.glob(os.path.join(data_dir, "*edges*.csv"))
             or glob.glob(os.path.join(data_dir, "*synapse*.csv")))
    if neurons and conns:
        return load_real_connectome(neurons[0], conns[0], **kwargs)
    return None


# --------------------------------------------------------------------------- #
# Synthetic fallback (Dale's law, biologically-plausible)
# --------------------------------------------------------------------------- #

def make_synthetic_connectome(
    n: int = 1200,
    density: float = 0.02,
    exc_fraction: float = 0.8,
    seed: int = 0,
) -> Connectome:
    """A sparse Dale's-law random network standing in for the real connectome.

    Not a fly -- but it has the structural features that matter for the model:
    sparse connectivity, an ~80/20 excitatory/inhibitory split, sign-consistent
    outputs per neuron (Dale's law), and heavy-tailed weights. Use it to
    develop and test, then swap in load_real_connectome()."""
    rng = np.random.default_rng(seed)
    n_exc = int(round(n * exc_fraction))
    sign = np.ones(n, dtype=np.float32)
    sign[n_exc:] = -1.0
    rng.shuffle(sign)

    nnz = int(density * n * n)
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]
    # Log-normal synaptic magnitudes; inhibitory a touch stronger (as in cortex).
    mag = rng.lognormal(mean=0.0, sigma=0.7, size=len(rows)).astype(np.float32)
    data = mag * sign[cols]                # sign comes from presynaptic neuron
    data[sign[cols] < 0] *= 1.3
    W = sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    W = W.tocsr()
    body_ids = np.arange(n)
    types = np.array(["exc" if s > 0 else "inh" for s in sign], dtype=object)
    return Connectome(
        W=W, sign=sign, body_ids=body_ids, types=types, source="synthetic",
        meta={"n_neurons": n, "density": density, "exc_fraction": exc_fraction, "seed": seed},
    )


def load_connectome(config: dict) -> Connectome:
    """Top-level entry point used by train.py.

    config keys (all optional):
        data_dir, min_synapses, max_neurons, keep_types,   # real
        synthetic_n, synthetic_density, seed,               # synthetic
        spectral_radius                                     # rescale
    Falls back to synthetic if no real CSVs are found."""
    conn = None
    data_dir = config.get("data_dir", "data/connectome")
    if config.get("use_real", True) and os.path.isdir(data_dir):
        try:
            conn = autoload_real(
                data_dir,
                min_synapses=config.get("min_synapses", 5),
                max_neurons=config.get("max_neurons", None),
                keep_types=config.get("keep_types", None),
            )
        except Exception as e:  # pragma: no cover - defensive
            print(f"[connectome] real load failed ({e}); using synthetic.")
    if conn is None:
        conn = make_synthetic_connectome(
            n=config.get("synthetic_n", 1200),
            density=config.get("synthetic_density", 0.02),
            exc_fraction=config.get("exc_fraction", 0.8),
            seed=config.get("seed", 0),
        )
    conn.rescale_spectral_radius(config.get("spectral_radius", 1.1))
    return conn


if __name__ == "__main__":
    c = make_synthetic_connectome(n=500)
    print(json.dumps({
        "source": c.source, "n": c.n, "density": round(c.density, 4),
        "spectral_radius": round(c.spectral_radius(), 3),
        "exc": int((c.sign > 0).sum()), "inh": int((c.sign < 0).sum()),
    }, indent=2))
