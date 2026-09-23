"""Velocity-field networks.

Deliberately plain MLPs: for fixed-dimension vector data (spectra, expression
profiles, latent codes, internal coordinates of a fixed-composition system) an
MLP is a genuinely competitive vector field, and it keeps the moving parts
visible. Swap this out for an equivariant network only once you actually have
Cartesian coordinates with symmetry to respect.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """Transformer-style sinusoidal features for the scalar path time t in [0, 1]."""

    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"time embedding dim must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        # t is scaled up so that the range [0, 1] actually spans the frequency band.
        args = t.float().view(-1, 1) * 1000.0 * freqs.view(1, -1)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    """Pre-norm residual MLP block, with time/condition injected as a FiLM shift."""

    def __init__(self, width: int, cond_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.cond = nn.Linear(cond_dim, 2 * width)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Start each block as an identity map so depth does not hurt early training.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        scale, shift = self.cond(c).chunk(2, dim=-1)
        x = self.norm(h) * (1.0 + scale) + shift
        x = self.drop(self.act(self.fc1(x)))
        x = self.fc2(x)
        return h + x


class VelocityMLP(nn.Module):
    """v_theta(x, t, y) -> velocity with the same shape as x.

    Args:
        data_dim:  dimension D of a flattened sample.
        cond_dim:  dimension of the conditioning vector y, or 0 for unconditional.
        width:     hidden width.
        depth:     number of residual blocks.
        time_dim:  width of the sinusoidal time features.
    """

    def __init__(
        self,
        data_dim: int,
        cond_dim: int = 0,
        width: int = 512,
        depth: int = 4,
        time_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.cond_dim = cond_dim

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.cond_embed = (
            nn.Sequential(nn.Linear(cond_dim, width), nn.SiLU(), nn.Linear(width, width))
            if cond_dim > 0
            else None
        )

        self.inp = nn.Linear(data_dim, width)
        self.blocks = nn.ModuleList(ResBlock(width, width, dropout) for _ in range(depth))
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, data_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        shape = x.shape
        h = self.inp(x.reshape(shape[0], -1))
        c = self.time_embed(t)
        if self.cond_embed is not None:
            if y is None:
                raise ValueError("model was built with cond_dim > 0 but y is None")
            c = c + self.cond_embed(y)
        for block in self.blocks:
            h = block(h, c)
        return self.out(self.out_norm(h)).reshape(shape)
