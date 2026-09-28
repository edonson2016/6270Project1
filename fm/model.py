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


@torch.no_grad()
def ot_pair(x0: Tensor, x1: Tensor) -> Tensor:
    """Reorder x0 so x0[k] is the OT partner of x1[k], within this minibatch.

    Minimizes sum_k ||x0[perm[k]] - x1[k]||^2 exactly via the Hungarian algorithm
    when SciPy is present, and greedily otherwise so the package keeps working on
    a torch+numpy-only install. Greedy is a decent approximation at batch 128 and
    is only ever a fallback.
    """
    a, b = x0.flatten(1), x1.flatten(1)
    C = torch.cdist(a, b).pow(2)
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as _np
        _, col = linear_sum_assignment(C.cpu().numpy())
        # col[i] is the target matched to source i; invert it to index x0 by target
        perm = torch.as_tensor(_np.argsort(col), device=x0.device)
    except ImportError:
        n = C.shape[0]
        order = torch.argsort(C.flatten())
        perm = torch.full((n,), -1, dtype=torch.long, device=x0.device)
        used_src, used_tgt = set(), set()
        for f in order.tolist():
            i, j = divmod(f, n)
            if i in used_src or j in used_tgt:
                continue
            perm[j] = i; used_src.add(i); used_tgt.add(j)
            if len(used_tgt) == n:
                break
    return x0[perm]


class FlowMatching(nn.Module):
    """Wraps a velocity network + an interpolant.

    Args:
        net: module with signature (x, t, y) -> velocity, shaped like x.
        path: the interpolant. Defaults to the standard linear one.
    """

    def __init__(self, net: nn.Module, path: CondOTPath | None = None, source=None,
                 coupling: str = "independent") -> None:
        super().__init__()
        self.net = net
        self.path = path or CondOTPath()
        # "independent" pairs each source draw with an unrelated data point, which
        # is the standard choice. "ot" solves an assignment within the minibatch so
        # each x0 is paired with a NEARBY x1.
        #
        # This is what makes a non-Gaussian source worth having. Under an
        # independent coupling the field must map arbitrary pairs, so moving p0
        # closer to p1 shrinks ||u_t|| without making its direction any more
        # predictable -- a worse-conditioned target, not a better one. OT coupling
        # is what turns proximity between the two distributions into short,
        # consistent displacements.
        if coupling not in ("independent", "ot"):
            raise ValueError(coupling)
        self.coupling = coupling
        # None = standard N(0, I). Anything with .sample(n, device) works; see
        # fm/source.py. The conditional target u_t = x1 - x0 is unchanged whatever
        # the source is, so this swaps cleanly -- but a source closer to the data
        # also does more of the generative work, which the `prior` arm measures.
        self.source = source

    def forward(self, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        """Evaluate the learned vector field. This is the ODE right-hand side."""
        return self.net(x, t, y)

    def sample_source(self, x1: Tensor) -> Tensor:
        """Source samples matched to x1's shape; N(0, I) unless a source was given."""
        if self.source is None:
            return torch.randn_like(x1)
        return self.source.sample(x1.shape[0], device=x1.device).to(x1.dtype)

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
            if self.coupling == "ot":
                x0 = ot_pair(x0, x1)
        t = self.path.sample_t(x1.shape[0], device=x1.device, dtype=x1.dtype)
        xt, ut = self.path.interpolate(x0, x1, t)
        vt = self.net(xt, t, y)
        per_sample = (vt - ut).pow(2).flatten(1).mean(dim=1)
        if reduction == "none":
            return per_sample
        if reduction == "sum":
            return per_sample.sum()
        return per_sample.mean()
