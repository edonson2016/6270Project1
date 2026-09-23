"""The flow matching model: a velocity field trained by regression onto the
conditional target velocity of the interpolant.

The whole objective is three lines (see `loss`). Everything else in this package
is plumbing around it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .paths import CondOTPath


class FlowMatching(nn.Module):
    """Wraps a velocity network + an interpolant.

    Args:
        net: module with signature (x, t, y) -> velocity, shaped like x.
        path: the interpolant. Defaults to the standard linear one.
    """

    def __init__(self, net: nn.Module, path: CondOTPath | None = None) -> None:
        super().__init__()
        self.net = net
        self.path = path or CondOTPath()

    def forward(self, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        """Evaluate the learned vector field. This is the ODE right-hand side."""
        return self.net(x, t, y)

    def sample_source(self, x1: Tensor) -> Tensor:
        """Default source distribution: standard Gaussian, matched to x1's shape."""
        return torch.randn_like(x1)

    def loss(
        self,
        x1: Tensor,
        y: Tensor | None = None,
        x0: Tensor | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        """Conditional flow matching loss.

        Args:
            x1: target samples, shape (B, ...). Your data.
            y:  optional conditioning, shape (B, cond_dim).
            x0: optional source samples with the same shape as x1. Leave as None
                for generation from noise. Pass paired samples to instead learn
                transport between two *data* distributions (e.g. control cells ->
                perturbed cells, unrelaxed geometry -> relaxed geometry). The
                pairing is whatever the dataloader hands you; an independent
                coupling is the standard choice and is what this defaults to.
        """
        if x0 is None:
            x0 = self.sample_source(x1)
        t = self.path.sample_t(x1.shape[0], device=x1.device, dtype=x1.dtype)
        xt, ut = self.path.interpolate(x0, x1, t)
        vt = self.net(xt, t, y)
        per_sample = (vt - ut).pow(2).flatten(1).mean(dim=1)
        if reduction == "none":
            return per_sample
        if reduction == "sum":
            return per_sample.sum()
        return per_sample.mean()
