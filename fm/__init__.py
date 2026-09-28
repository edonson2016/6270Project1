"""A minimal, readable flow matching implementation.

Standard linear interpolant, no guidance, plain ODE vector field.
"""

from .paths import CondOTPath
from .nets import VelocityMLP, SinusoidalTimeEmbedding, make_velocity_net
from .model import FlowMatching
from .ddpm import DDPM, linear_beta_schedule, cosine_beta_schedule, n_function_evals
from .sampling import sample, integrate
from .ema import EMA
from .data import Standardizer, VectorDataset, PairedVectorDataset
from .train import Trainer, TrainConfig
from .data import CFGVectorDataset
from .guidance import CFGVelocity, AutoGuidedVelocity, guided_nfe

__all__ = [
    "CondOTPath",
    "VelocityMLP",
    "SinusoidalTimeEmbedding",
    "FlowMatching",
    "DDPM",
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "n_function_evals",
    "sample",
    "integrate",
    "EMA",
    "Standardizer",
    "VectorDataset",
    "PairedVectorDataset",
    "Trainer",
    "TrainConfig",
]
