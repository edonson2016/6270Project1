"""A plain MLP autoencoder for fixed-dimension continuous data.

Used for the conformer experiment, where the data is already continuous and the
question is purely about dimensionality: how small a latent can you compress a
molecular conformer into before flow matching in that latent stops reproducing
the physics?

Deterministic by default (beta=0). For continuous data with a flow model on top,
the KL term's usual job -- making the latent samplable -- is exactly what flow
matching takes over, so there is no reason to pay for it in reconstruction. Set
beta > 0 to compare against the VAE regime.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MLPAutoencoder(nn.Module):
    def __init__(self, data_dim: int, latent_dim: int, width: int = 256,
                 depth: int = 3, beta: float = 0.0) -> None:
        super().__init__()
        self.data_dim, self.latent_dim, self.beta = data_dim, latent_dim, beta

        def stack(d_in, d_out):
            layers, d = [], d_in
            for _ in range(depth):
                layers += [nn.Linear(d, width), nn.SiLU()]
                d = width
            layers += [nn.Linear(d, d_out)]
            return nn.Sequential(*layers)

        self.encoder = stack(data_dim, latent_dim * (2 if beta > 0 else 1))
        self.decoder = stack(latent_dim, data_dim)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        h = self.encoder(x)
        if self.beta > 0:
            mu, logvar = h.chunk(2, dim=-1)
            return mu, logvar.clamp(-8, 8)
        return h, None

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> dict:
        mu, logvar = self.encode(x)
        if logvar is not None and self.training:
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z = mu
        xh = self.decode(z)
        recon = (xh - x).pow(2).mean()
        out = {"recon": recon, "loss": recon}
        if logvar is not None:
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)).mean()
            out["kl"] = kl
            out["loss"] = recon + self.beta * kl
        return out
