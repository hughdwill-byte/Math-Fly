"""
Sensory encoding and motor decoding for the fly network.

A connectome is just neurons and wires; to make it *do math* we have to decide
how a math problem is presented to it (which neurons get driven, and how) and
how an answer is read back out (which neurons we listen to). This mirrors real
neuroscience: a small set of "sensory" neurons receive stimulus-driven input,
and a small set of "motor"/readout neurons report the decision.

We support the encoding scheme that flies actually seem to use for small
quantities -- a mixture of a **place code** (a dedicated input channel per
token) and a **magnitude code** (activity proportional to the number, the way
numerosity-tuned neurons behave). Problems are delivered as a short *sequence*
of token pulses (e.g. `3`, `+`, `4`) so the recurrent dynamics have to hold and
combine information over time -- that temporal integration is the actual
computation.
"""

from __future__ import annotations

import numpy as np

# Token vocabulary. Digits 0-9 plus operators and control tokens.
OPERATORS = ["+", "-", "*", "=", "<PAD>", "<GO>"]
DIGITS = [str(d) for d in range(10)]
VOCAB = DIGITS + OPERATORS
TOKEN2ID = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


def tokenize_problem(a: int, op: str, b: int) -> list[str]:
    """Render `a op b =` as a token sequence, most-significant digit first."""
    seq = list(str(a)) + [op] + list(str(b)) + ["="]
    return seq


class IOEncoder:
    """Maps token sequences to per-timestep input vectors over input neurons,
    and maps readout-neuron activity to an answer via a trained readout.

    The encoder owns *which* neurons are sensory/motor and the fixed random
    (but frozen) projection from tokens into those neurons. Only the readout
    weights are learned (see model.py / train.py)."""

    def __init__(
        self,
        n_neurons: int,
        n_input: int = 64,
        n_readout: int = 128,
        n_answers: int = 100,
        steps_per_token: int = 4,
        input_gain: float = 1.0,
        n_digits: int = 4,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.n_neurons = n_neurons
        self.n_input = min(n_input, n_neurons)
        self.n_readout = min(n_readout, n_neurons)
        self.n_answers = n_answers
        self.n_digits = n_digits          # for digit-serial decoding
        self.steps_per_token = steps_per_token
        self.input_gain = input_gain

        # Disjoint sensory and motor populations.
        perm = rng.permutation(n_neurons)
        self.input_neurons = perm[: self.n_input]
        self.readout_neurons = perm[self.n_input : self.n_input + self.n_readout]

        # Frozen embedding: each token -> a sparse random pattern over sensory
        # neurons (place code), scaled by magnitude for digits (magnitude code).
        self.embed = rng.normal(0, 1, size=(VOCAB_SIZE, self.n_input)).astype(np.float32)
        self.embed *= (rng.random((VOCAB_SIZE, self.n_input)) < 0.5)  # 50% sparse
        # Add a graded magnitude channel for digits: extra drive on a shared
        # "numerosity" sub-population proportional to the digit's value.
        mag_channel = rng.permutation(self.n_input)[: max(4, self.n_input // 8)]
        for d in range(10):
            self.embed[TOKEN2ID[str(d)], mag_channel] += 0.3 * d

    # -- sensory --------------------------------------------------------- #
    def encode(self, tokens: list[str]) -> np.ndarray:
        """Return (T, n_neurons) input current time-series for a token seq.
        Each token is held for `steps_per_token` steps; a final answer/settle
        window of GO pulses follows so the net can compute before we read it."""
        T = (len(tokens) + self.settle_tokens) * self.steps_per_token
        U = np.zeros((T, self.n_neurons), dtype=np.float32)
        t = 0
        for tok in tokens:
            vec = self.embed[TOKEN2ID[tok]] * self.input_gain
            for _ in range(self.steps_per_token):
                U[t, self.input_neurons] = vec
                t += 1
        # Settle window: GO token drive while the readout is taken.
        go = self.embed[TOKEN2ID["<GO>"]] * self.input_gain
        for _ in range(self.settle_tokens * self.steps_per_token):
            U[t, self.input_neurons] = go
            t += 1
        return U

    settle_tokens = 3  # how long the network "thinks" after seeing "="

    def readout_window(self, T: int) -> slice:
        """Timesteps whose states are averaged to form the answer readout
        (the settle window at the end)."""
        w = self.settle_tokens * self.steps_per_token
        return slice(T - w, T)

    # -- motor: single-class readout ------------------------------------- #
    def target_onehot(self, answer: int) -> np.ndarray:
        y = np.zeros(self.n_answers, dtype=np.float32)
        y[int(np.clip(answer, 0, self.n_answers - 1))] = 1.0
        return y

    def decode(self, logits: np.ndarray) -> int:
        return int(np.argmax(logits))

    # -- motor: digit-serial readout ------------------------------------- #
    # Instead of one big classifier over every possible answer (which explodes
    # for multi-digit results), we read out each decimal digit independently
    # with its own 10-way head. The answer space is then size-INDEPENDENT: a
    # 4-digit reader covers 0..9999 with just 4*10 outputs. Position 0 = units.
    def target_digits(self, answer: int) -> np.ndarray:
        """Return (n_digits,) array of digit classes 0-9, units-first."""
        a = int(max(0, answer))
        out = np.empty(self.n_digits, dtype=np.int64)
        for i in range(self.n_digits):
            out[i] = a % 10
            a //= 10
        return out

    def target_digits_onehot(self, answer: int) -> np.ndarray:
        """(n_digits, 10) one-hot targets for the digit heads."""
        digits = self.target_digits(answer)
        Y = np.zeros((self.n_digits, 10), dtype=np.float32)
        Y[np.arange(self.n_digits), digits] = 1.0
        return Y

    def decode_digits(self, digit_logits: np.ndarray) -> int:
        """digit_logits: (n_digits, 10) -> integer. Position 0 = units."""
        preds = np.argmax(digit_logits, axis=-1)
        return int(sum(int(d) * (10 ** i) for i, d in enumerate(preds)))
