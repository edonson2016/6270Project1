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


# ---------------------------------------------------------------- architecture ablation
# Three alternatives to VelocityMLP, all with the same (x, t, y) -> v interface so they
# drop into FlowMatching and DDPM unchanged. Defaults are sized to match VelocityMLP's
# ~2.58M parameters at data_dim=16, so a comparison changes the architecture and not
# the capacity.
#
# The conv models read the D latent coordinates as a length-D, 1-channel sequence.
# That imposes locality and weight sharing across coordinates the latent does not
# have -- MLP_down emits them in arbitrary order -- so they are negative controls for
# that inductive bias, not contenders.


def _time_mlp(time_dim: int, width: int) -> nn.Module:
    return nn.Sequential(SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, width),
                         nn.SiLU(), nn.Linear(width, width))


class FFNN(nn.Module):
    """Plain feed-forward baseline: [x, time features, y] concatenated at the input.

    No residuals, no normalization, no FiLM, default initialization -- the textbook
    network, to test whether VelocityMLP's structure matters at all. Time enters only
    once, at the first layer.
    """

    def __init__(self, data_dim: int, cond_dim: int = 0, width: int = 896,
                 depth: int = 4, time_dim: int = 128) -> None:
        super().__init__()
        self.data_dim, self.cond_dim = data_dim, cond_dim
        self.time = SinusoidalTimeEmbedding(time_dim)
        layers, d_in = [], data_dim + time_dim + cond_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.SiLU()]
            d_in = width
        layers.append(nn.Linear(width, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        parts = [x.reshape(x.shape[0], -1), self.time(t)]
        if self.cond_dim:
            if y is None:
                raise ValueError("model was built with cond_dim > 0 but y is None")
            parts.append(y)
        return self.net(torch.cat(parts, -1)).reshape(x.shape)


class ConvResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv1d, FiLM from the time/condition vector, twice, residual."""

    def __init__(self, cin: int, cout: int, cond_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, cin)
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * cout)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.film(c).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        return self.skip(x) + self.conv2(self.act(h))


class _ConvBase(nn.Module):
    def _setup_cond(self, data_dim, cond_dim, cdim, time_dim):
        self.data_dim, self.cond_dim = data_dim, cond_dim
        self.time_embed = _time_mlp(time_dim, cdim)
        self.cond_embed = (nn.Sequential(nn.Linear(cond_dim, cdim), nn.SiLU(),
                                         nn.Linear(cdim, cdim)) if cond_dim > 0 else None)

    def _cond(self, t: Tensor, y: Tensor | None) -> Tensor:
        c = self.time_embed(t)
        if self.cond_embed is not None:
            if y is None:
                raise ValueError("model was built with cond_dim > 0 but y is None")
            c = c + self.cond_embed(y)
        return c


class Conv1dNet(_ConvBase):
    """Residual 1-D CNN over the D latent coordinates, FiLM time conditioning."""

    def __init__(self, data_dim: int, cond_dim: int = 0, channels: int = 280,
                 depth: int = 4, time_dim: int = 128) -> None:
        super().__init__()
        self._setup_cond(data_dim, cond_dim, channels, time_dim)
        self.inp = nn.Conv1d(1, channels, 3, padding=1)
        self.blocks = nn.ModuleList(ConvResBlock(channels, channels, channels)
                                    for _ in range(depth))
        self.out_norm = nn.GroupNorm(8, channels)
        self.out = nn.Conv1d(channels, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        c = self._cond(t, y)
        h = self.inp(x.reshape(x.shape[0], 1, -1))
        for b in self.blocks:
            h = b(h, c)
        return self.out(torch.nn.functional.silu(self.out_norm(h))).reshape(x.shape)


class UNet1d(_ConvBase):
    """Two-level 1-D U-Net (D -> D/2 -> D/4 and back) with skip connections.

    The standard diffusion backbone, reduced to vectors. Needs D divisible by 4.
    """

    def __init__(self, data_dim: int, cond_dim: int = 0, channels: int = 112,
                 time_dim: int = 128) -> None:
        super().__init__()
        if data_dim % 4:
            raise ValueError(f"UNet1d needs data_dim divisible by 4, got {data_dim}")
        C, C2 = channels, 2 * channels
        self._setup_cond(data_dim, cond_dim, C2, time_dim)
        self.inp = nn.Conv1d(1, C, 3, padding=1)
        self.d1 = ConvResBlock(C, C, C2)
        self.down1 = nn.Conv1d(C, C, 3, stride=2, padding=1)
        self.d2 = ConvResBlock(C, C2, C2)
        self.down2 = nn.Conv1d(C2, C2, 3, stride=2, padding=1)
        self.m1 = ConvResBlock(C2, C2, C2)
        self.m2 = ConvResBlock(C2, C2, C2)
        self.up2 = nn.Conv1d(C2, C2, 3, padding=1)
        self.u2 = ConvResBlock(2 * C2, C2, C2)
        self.up1 = nn.Conv1d(C2, C2, 3, padding=1)
        self.u1 = ConvResBlock(C2 + C, C, C2)
        self.out_norm = nn.GroupNorm(8, C)
        self.out = nn.Conv1d(C, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        up = torch.nn.functional.interpolate
        c = self._cond(t, y)
        h = self.inp(x.reshape(x.shape[0], 1, -1))
        s1 = self.d1(h, c)                                   # D
        s2 = self.d2(self.down1(s1), c)                      # D/2
        h = self.m2(self.m1(self.down2(s2), c), c)           # D/4
        h = self.up2(up(h, scale_factor=2, mode="nearest"))  # D/2
        h = self.u2(torch.cat([h, s2], 1), c)
        h = self.up1(up(h, scale_factor=2, mode="nearest"))  # D
        h = self.u1(torch.cat([h, s1], 1), c)
        return self.out(torch.nn.functional.silu(self.out_norm(h))).reshape(x.shape)


def make_velocity_net(arch: str, data_dim: int, cond_dim: int = 0,
                      width: int = 384) -> nn.Module:
    """`width` applies to the default 'mlp' only; the others use their matched defaults."""
    if arch == "mlp":
        return VelocityMLP(data_dim, cond_dim=cond_dim, width=width, depth=4)
    if arch == "ffnn":
        return FFNN(data_dim, cond_dim=cond_dim)
    if arch == "cnn":
        return Conv1dNet(data_dim, cond_dim=cond_dim)
    if arch == "unet":
        return UNet1d(data_dim, cond_dim=cond_dim)
    raise ValueError(f"unknown fm arch {arch!r}")
