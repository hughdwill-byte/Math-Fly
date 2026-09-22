"""Math-Fly: train a Drosophila connectome-constrained network to do math.

Public API:
    load_connectome, make_synthetic_connectome   -- build the recurrent core
    IOEncoder                                     -- sensory/motor interface
    ReservoirModel, build_torch_rnn               -- the models
    CURRICULUM                                    -- the math task ladder
    train_reservoir, train_bptt                   -- training entry points
"""

from .connectome import (Connectome, load_connectome, load_real_connectome,
                         make_synthetic_connectome)
from .io_encoding import IOEncoder, VOCAB, tokenize_problem
from .model import ReservoirModel, build_torch_rnn
from .envs import CURRICULUM, CURRICULUM_BY_NAME, MathTask, make_dataset
from .train import train_reservoir, train_bptt

__version__ = "0.1.0"

__all__ = [
    "Connectome", "load_connectome", "load_real_connectome",
    "make_synthetic_connectome", "IOEncoder", "VOCAB", "tokenize_problem",
    "ReservoirModel", "build_torch_rnn", "CURRICULUM", "CURRICULUM_BY_NAME",
    "MathTask", "make_dataset", "train_reservoir", "train_bptt", "__version__",
]
