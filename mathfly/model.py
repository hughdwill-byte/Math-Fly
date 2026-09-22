"""
The fly network: a continuous-time rate model whose recurrent connectivity IS
the connectome.

Dynamics (leaky rate neurons, Euler-integrated):

    x[t+1] = (1 - alpha) * x[t] + alpha * ( W_rec @ r[t] + W_in @ u[t] + b )
    r[t]   = phi(x[t])            phi = tanh   (bounded firing rate)

W_rec is FIXED by the connectome (sign + magnitude of every edge). This is the
biological prior: the network can only route information along wires the fly
actually has. We provide two training regimes:

  * ReservoirModel  (NumPy, no autograd): freeze W_rec entirely and train only
    a linear readout W_out from the motor neurons. This is Reservoir Computing
    / Echo-State: fast, CPU-friendly, always available, and a legitimate model
    of how a fixed recurrent circuit can support flexible downstream learning
    (cf. cerebellum / mushroom-body readout learning in the fly).

  * TorchRNN  (PyTorch autograd): keep the connectome's SIGN and SPARSITY
    fixed but let a per-edge gain, the input weights and the readout be learned
    by backprop-through-time. This can reach harder tasks but needs more
    compute. Dale's law is enforced after every step.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .connectome import Connectome
from .io_encoding import IOEncoder


# --------------------------------------------------------------------------- #
# NumPy reservoir model
# --------------------------------------------------------------------------- #

class ReservoirModel:
    """Fixed connectome reservoir + trained linear readout."""

    def __init__(self, connectome: Connectome, encoder: IOEncoder,
                 alpha: float = 0.3, noise: float = 0.0, seed: int = 0):
        self.C = connectome
        self.enc = encoder
        self.alpha = alpha
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.W = connectome.W                      # (N, N) sparse, fixed
        self.b = np.zeros(self.C.n, dtype=np.float32)
        # Readout: from motor-neuron rates -> answer logits. Trained.
        # The reservoir (connectome) is shared across tasks; each task gets its
        # own readout head, stored by name in `self.readouts`. `W_out` is the
        # currently-selected head.
        self.W_out = np.zeros((encoder.n_answers, encoder.n_readout + 1), dtype=np.float32)
        self.readouts: dict[str, np.ndarray] = {}
        # Digit-serial heads: name -> (n_digits, 10, n_readout+1) weight stack.
        self.digit_heads: dict[str, np.ndarray] = {}

    def run(self, U: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """Integrate the dynamics for input current series U (T, N).
        Returns the state trajectory R (T, N) of firing rates."""
        N = self.C.n
        x = np.zeros(N, dtype=np.float32) if x0 is None else x0.copy()
        T = U.shape[0]
        R = np.empty((T, N), dtype=np.float32)
        W = self.W
        a = self.alpha
        for t in range(T):
            r = np.tanh(x)
            rec = W.dot(r)
            x = (1 - a) * x + a * (rec + U[t] + self.b)
            if self.noise:
                x += self.noise * self.rng.standard_normal(N).astype(np.float32)
            R[t] = np.tanh(x)
        return R

    def features(self, tokens: list[str]) -> np.ndarray:
        """Readout feature vector for one problem: mean motor-neuron rate over
        the settle window, plus a bias term."""
        U = self.enc.encode(tokens)
        R = self.run(U)
        win = self.enc.readout_window(R.shape[0])
        feat = R[win][:, self.enc.readout_neurons].mean(axis=0)
        return np.concatenate([feat, [1.0]]).astype(np.float32)  # +bias

    def select(self, name: str):
        """Point W_out at a named, previously-trained readout head."""
        if name not in self.readouts:
            raise KeyError(f"no readout trained for task '{name}'; have {list(self.readouts)}")
        self.W_out = self.readouts[name]
        return self

    def logits(self, tokens: list[str], name: str | None = None) -> np.ndarray:
        W = self.readouts[name] if name is not None else self.W_out
        return W @ self.features(tokens)

    def predict(self, tokens: list[str], name: str | None = None) -> int:
        return int(np.argmax(self.logits(tokens, name)))

    # -- training: ridge regression on collected features ---------------- #
    def fit_readout(self, feats: np.ndarray, targets: np.ndarray, ridge: float = 1e-2,
                    name: str | None = None):
        """Closed-form ridge regression: W_out = Y^T X (X^T X + lambda I)^-1.
        Stores the head under `name` (if given) and selects it."""
        X = feats                                  # (M, F)
        Y = targets                                # (M, A) one-hot
        F = X.shape[1]
        A = X.T @ X + ridge * np.eye(F, dtype=np.float32)
        B = X.T @ Y
        self.W_out = np.linalg.solve(A, B).T.astype(np.float32)
        if name is not None:
            self.readouts[name] = self.W_out
        return self.W_out

    def partial_fit_readout(self, feats, targets, lr=0.05):
        """Online least-squares step (for streaming curricula)."""
        for x, y in zip(feats, targets):
            pred = self.W_out @ x
            self.W_out += lr * np.outer(y - pred, x).astype(np.float32)

    # -- digit-serial training / prediction ------------------------------ #
    def fit_digit_readout(self, feats: np.ndarray, answers, ridge: float = 1e-2,
                          name: str | None = None):
        """Fit one 10-way ridge classifier per digit position from the same
        reservoir features. `answers` is a 1-D array of integer answers.
        Stores a (n_digits, 10, F) weight stack under `name`."""
        D = self.enc.n_digits
        F = feats.shape[1]
        A = feats.T @ feats + ridge * np.eye(F, dtype=np.float32)
        Ainv_Xt = np.linalg.solve(A, feats.T)               # (F, M), shared
        stack = np.zeros((D, 10, F), dtype=np.float32)
        Yall = np.stack([self.enc.target_digits_onehot(a) for a in answers])  # (M, D, 10)
        for d in range(D):
            Wd = (Ainv_Xt @ Yall[:, d, :]).T                # (10, F)
            stack[d] = Wd.astype(np.float32)
        if name is not None:
            self.digit_heads[name] = stack
        self._cur_digit = stack
        return stack

    def predict_number(self, tokens: list[str], name: str | None = None) -> int:
        stack = self.digit_heads[name] if name is not None else self._cur_digit
        x = self.features(tokens)
        digit_logits = stack @ x                             # (n_digits, 10)
        return self.enc.decode_digits(digit_logits)

    # -- persistence ----------------------------------------------------- #
    def save(self, path: str):
        heads = {f"head::{k}": v for k, v in self.readouts.items()}
        dheads = {f"digit::{k}": v for k, v in self.digit_heads.items()}
        np.savez(path, W_out=self.W_out, b=self.b, alpha=np.float32(self.alpha),
                 **heads, **dheads)

    def load(self, path: str):
        d = np.load(path)
        self.W_out = d["W_out"]; self.b = d["b"]; self.alpha = float(d["alpha"])
        self.readouts = {k[len("head::"):]: d[k] for k in d.files if k.startswith("head::")}
        self.digit_heads = {k[len("digit::"):]: d[k] for k in d.files if k.startswith("digit::")}


# --------------------------------------------------------------------------- #
# Optional PyTorch backprop-through-time model
# --------------------------------------------------------------------------- #

def build_torch_rnn(connectome: Connectome, encoder: IOEncoder, **kw):
    """Construct a TorchRNN. Imported lazily so NumPy-only installs still work."""
    import torch
    import torch.nn as nn

    class TorchRNN(nn.Module):
        """Connectome-constrained trainable RNN. The connectome fixes which
        edges exist and their sign; a positive per-edge gain is learned, so
        Dale's law and the wiring diagram are preserved throughout training."""

        def __init__(self, C: Connectome, enc: IOEncoder, alpha=0.3, seed=0):
            super().__init__()
            torch.manual_seed(seed)
            self.enc = enc
            self.alpha = alpha
            self.N = C.n
            coo = C.W.tocoo()
            idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
            self.register_buffer("edge_index", idx)
            self.register_buffer("edge_sign", torch.tensor(np.sign(coo.data), dtype=torch.float32))
            # Learnable log-gain per edge (>=0 magnitude via softplus).
            init = np.log(np.expm1(np.abs(coo.data) + 1e-3)).astype(np.float32)
            self.edge_loggain = nn.Parameter(torch.tensor(init))
            self.register_buffer("in_neurons", torch.tensor(enc.input_neurons, dtype=torch.long))
            self.register_buffer("out_neurons", torch.tensor(enc.readout_neurons, dtype=torch.long))
            self.b = nn.Parameter(torch.zeros(self.N))
            self.readout = nn.Linear(enc.n_readout, enc.n_answers)

        def _sparse_W(self):
            import torch
            import torch.nn.functional as F
            vals = self.edge_sign * F.softplus(self.edge_loggain)
            return torch.sparse_coo_tensor(self.edge_index, vals, (self.N, self.N)).coalesce()

        def forward(self, U):  # U: (B, T, N)
            import torch
            W = self._sparse_W()
            B, T, N = U.shape
            x = torch.zeros(B, N, device=U.device)
            a = self.alpha
            win = self.enc.readout_window(T)
            acc = []
            for t in range(T):
                r = torch.tanh(x)
                rec = torch.sparse.mm(W, r.t()).t()        # (B, N)
                x = (1 - a) * x + a * (rec + U[:, t, :] + self.b)
                if t >= win.start:
                    acc.append(torch.tanh(x)[:, self.out_neurons])
            feat = torch.stack(acc, 0).mean(0)             # (B, n_readout)
            return self.readout(feat)

    return TorchRNN(connectome, encoder, **kw)
