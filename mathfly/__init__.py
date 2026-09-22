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
from .train import train_reservoir, train_reservoir_digits, train_bptt

__version__ = "0.1.0"

__all__ = [
    "Connectome", "load_connectome", "load_real_connectome",
    "make_synthetic_connectome", "IOEncoder", "VOCAB", "tokenize_problem",
    "ReservoirModel", "build_torch_rnn", "CURRICULUM", "CURRICULUM_BY_NAME",
    "MathTask", "make_dataset", "train_reservoir", "train_reservoir_digits",
    "train_bptt", "__version__",
]


def make_env(*args, **kwargs):
    """Lazy re-export of the Gymnasium env builder (keeps gymnasium optional)."""
    from .gym_env import make_env as _make_env
    return _make_env(*args, **kwargs)


def build_male_cns_subgraph(*args, **kwargs):
    """Lazy re-export: build a real male-CNS connectome subgraph (needs pyarrow)."""
    from .male_cns import build_subgraph
    return build_subgraph(*args, **kwargs)


def train_synapses(*args, **kwargs):
    """Lazy re-export: synapse-training (BPTT) on the real connectome (needs torch)."""
    from .synapse_train import train
    return train(*args, **kwargs)


def CalcFly(*args, **kwargs):
    """Lazy re-export: the calculator fly (real connectome -> single-digit maths)."""
    from .calc_fly import CalcFly as _C
    return _C(*args, **kwargs)


def SciFly(*args, **kwargs):
    """Lazy re-export: the scientific-calculator fly (real connectome, 0-99)."""
    from .sci_fly import SciFly as _S
    return _S(*args, **kwargs)
