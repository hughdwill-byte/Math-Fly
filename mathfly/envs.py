"""
The "virtual environment" the fly lives in: a curriculum of arithmetic tasks.

Each task is a generator of (token_sequence, answer) pairs. The network sees
the token sequence as timed sensory input and must produce the answer at its
motor readout. Difficulty is layered so training can proceed from innate,
fly-plausible competences (comparing/counting small quantities) up to
multi-digit symbolic arithmetic -- letting you empirically discover how far a
connectome-constrained network can be pushed.

Fruit flies demonstrably do the low levels of this in the real world:
numerosity discrimination and "less/more" comparison of small sets. The higher
symbolic levels are beyond a real fly, so treat them as a stress test of the
architecture rather than a claim about the animal.
"""

from __future__ import annotations

import numpy as np

from .io_encoding import tokenize_problem


class MathTask:
    """One difficulty level. Generates problems and their answers."""

    def __init__(self, name: str, gen, max_answer: int, description: str = ""):
        self.name = name
        self._gen = gen
        self.max_answer = max_answer
        self.description = description

    def sample(self, rng) -> tuple[list[str], int]:
        return self._gen(rng)


def _compare(rng):
    a, b = rng.integers(0, 10), rng.integers(0, 10)
    # answer: 1 if a>b else 0  (encoded as a tiny classification)
    return tokenize_problem(a, "-", b), int(a > b)


def _count(rng):
    # "count" = sum of a sequence of 1s presented as a+a+...; here a+0 repeated
    # Simplify to identity-ish counting: report how many nonzero pulses (0..9)
    n = rng.integers(0, 10)
    return list(str(n)) + ["="], int(n)


def _add(lo, hi):
    def g(rng):
        a, b = rng.integers(lo, hi + 1), rng.integers(lo, hi + 1)
        return tokenize_problem(a, "+", b), int(a + b)
    return g


def _sub(lo, hi):
    def g(rng):
        a, b = rng.integers(lo, hi + 1), rng.integers(lo, hi + 1)
        if b > a:
            a, b = b, a               # keep answers non-negative
        return tokenize_problem(a, "-", b), int(a - b)
    return g


def _mul(lo, hi):
    def g(rng):
        a, b = rng.integers(lo, hi + 1), rng.integers(lo, hi + 1)
        return tokenize_problem(a, "*", b), int(a * b)
    return g


def _mixed(lo, hi):
    ops = ["+", "-", "*"]
    def g(rng):
        a, b = rng.integers(lo, hi + 1), rng.integers(lo, hi + 1)
        op = ops[rng.integers(0, 3)]
        if op == "-" and b > a:
            a, b = b, a
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        return tokenize_problem(a, op, b), int(ans)
    return g


# The ordered curriculum. Each entry escalates difficulty; max_answer sets the
# size of the classification readout needed at that level.
CURRICULUM = [
    MathTask("compare",    _compare,      max_answer=2,   description="is a > b? (fly-plausible)"),
    MathTask("count",      _count,        max_answer=10,  description="report a small quantity 0-9"),
    MathTask("add_1digit", _add(0, 9),    max_answer=18,  description="a + b, single digit"),
    MathTask("sub_1digit", _sub(0, 9),    max_answer=9,   description="a - b, non-negative"),
    MathTask("add_2digit", _add(0, 49),   max_answer=98,  description="a + b up to 49+49"),
    MathTask("mul_1digit", _mul(0, 9),    max_answer=81,  description="a * b, single digit"),
    MathTask("mixed_easy", _mixed(0, 9),  max_answer=81,  description="mixed +,-,* single digit"),
    MathTask("add_3digit", _add(0, 199),  max_answer=398, description="a + b up to 199+199"),
    MathTask("mixed_hard", _mixed(0, 20), max_answer=400, description="mixed ops up to 20"),
]

CURRICULUM_BY_NAME = {t.name: t for t in CURRICULUM}


def make_dataset(task: MathTask, n: int, seed: int = 0):
    """Sample n unique-ish (tokens, answer) pairs from a task."""
    rng = np.random.default_rng(seed)
    problems, answers = [], []
    for _ in range(n):
        toks, ans = task.sample(rng)
        problems.append(toks)
        answers.append(ans)
    return problems, np.array(answers, dtype=np.int64)
