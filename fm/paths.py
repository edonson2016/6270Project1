"""Probability paths (interpolants) for flow matching.

Only the standard / "conditional OT" (linear) interpolant is implemented, which
is what rectified flow and Lipman et al.'s CondOT path both reduce to:

    x_t = (1 - t) * x_0 + t * x_1
    u_t = x_1 - x_0                 (constant in t)

Convention used everywhere in this package:
    t = 0  ->  source / prior  (x_0)
    t = 1  ->  target / data   (x_1)
"""

from __future__ import annotations

import torch
from torch import Tensor


def expand_t(t: Tensor, x: Tensor) -> Tensor:
    """Reshape a per-sample t of shape (B,) to broadcast against x of shape (B, ...)."""
    return t.view(-1, *([1] * (x.dim() - 1)))


class CondOTPath:
    """The standard linear interpolant.

    Args:
        sigma_min: optional constant noise added around the interpolant. 0.0 gives
            the pure rectified-flow / deterministic interpolant, which is the
            default and the one to use unless you have a reason not to.
    """

    def __init__(self, sigma_min: float = 0.0) -> None:
        self.sigma_min = float(sigma_min)

    def sample_t(self, batch_size: int, device, dtype=torch.float32) -> Tensor:
        return torch.rand(batch_size, device=device, dtype=dtype)

    def interpolate(self, x0: Tensor, x1: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return (x_t, u_t): the point on the path and the target velocity there."""
        tt = expand_t(t, x0)
        xt = (1.0 - tt) * x0 + tt * x1
        ut = x1 - x0
        if self.sigma_min > 0.0:
            # Smoothing the path leaves the *conditional* target velocity unchanged;
            # it only widens the region of x-space the network is trained on.
            xt = xt + self.sigma_min * torch.randn_like(xt)
        return xt, ut
