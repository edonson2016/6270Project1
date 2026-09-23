"""Analytic FLOP accounting.

Deliberately simple and explicit, because the point is a fair comparison between
two arms, not an absolute number. The same accounting is applied to both, so the
Pareto frontier is meaningful even though the absolute counts are approximations.

Conventions (stated so they can be checked):
  - a Linear(in, out) forward on batch B costs 2*B*in*out FLOPs (multiply + add)
  - backward costs ~2x forward, so training a step is ~3x a forward pass
  - elementwise ops (SiLU, LayerNorm, residual adds) are ignored: they are O(B*width)
    against O(B*width^2) for the matmuls, well under 1% here
  - a GRU cell is 3 gates, each a Linear(in+hidden, hidden)
"""

from __future__ import annotations

import torch.nn as nn


def linear_flops(module: nn.Linear, batch: int = 1) -> int:
    return 2 * batch * module.in_features * module.out_features


def gru_flops(module: nn.GRU, batch: int, seq_len: int) -> int:
    """3 gates per step per layer, each a matmul over (input + hidden)."""
    total = 0
    h = module.hidden_size
    dirs = 2 if module.bidirectional else 1
    for layer in range(module.num_layers):
        inp = module.input_size if layer == 0 else h * dirs
        # per timestep: 3 * [ (inp x h) + (h x h) ] matmuls
        per_step = 2 * ((inp * h) + (h * h)) * 3
        total += per_step * seq_len * batch * dirs
    return total


def embedding_flops(module: nn.Embedding, batch: int, seq_len: int) -> int:
    return 0  # a lookup, not arithmetic


def forward_flops(model: nn.Module, batch: int = 1, seq_len: int = 1) -> int:
    """Sum forward FLOPs over the modules we care about."""
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # Linear layers inside a sequence model apply at every position.
            total += linear_flops(m, batch)
        elif isinstance(m, nn.GRU):
            total += gru_flops(m, batch, seq_len)
    return total


def seq_forward_flops(model: nn.Module, batch: int, seq_len: int,
                      per_position_linears: bool = True) -> int:
    """Forward FLOPs for a sequence model, where Linear heads act per position."""
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += linear_flops(m, batch * (seq_len if per_position_linears else 1))
        elif isinstance(m, nn.GRU):
            total += gru_flops(m, batch, seq_len)
    return total


def train_flops(fwd_flops_per_sample: int, n_samples_processed: int) -> int:
    """Training cost: forward + backward ~= 3x forward, per sample seen."""
    return 3 * fwd_flops_per_sample * n_samples_processed


def ode_sampling_flops(velocity_net: nn.Module, n_steps: int, method: str = "heun") -> int:
    """FLOPs to generate ONE sample by integrating the ODE.

    Heun is 2 network evaluations per step, Euler 1, RK4 4.
    """
    evals = {"euler": 1, "heun": 2, "rk4": 4}[method]
    return forward_flops(velocity_net, batch=1) * n_steps * evals


def sampling_flops(net: nn.Module, n_function_evals: int) -> int:
    """FLOPs to generate ONE sample, given a count of network evaluations.

    The honest unit for comparing an ODE sampler against a diffusion chain: both
    pay exactly one forward pass per evaluation, so the whole efficiency
    difference is the NFE count. Heun at 50 steps is 100; DDPM ancestral at
    T=1000 is 1000; DDIM at 50 steps is 50.
    """
    return forward_flops(net, batch=1) * n_function_evals


def fmt(n: float) -> str:
    for unit, div in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n/div:.2f}{unit}"
    return f"{n:.0f}"
