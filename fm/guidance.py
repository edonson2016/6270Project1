"""Guided velocity fields for sampling.

Both methods here are the same algebraic move -- evaluate two velocity fields
and extrapolate along their difference -- but they differ in what the second
field is, and that difference decides whether they work.

  classifier-free   v(x,t,null) + w*(v(x,t,y) - v(x,t,null))
                    one network, two conditionings. w=0 is unconditional,
                    w=1 is plain conditional generation, w>1 amplifies the
                    label's influence at the cost of diversity.

  autoguidance      (1+w)*v1(x,t) - w*v0(x,t)
                    two networks, no conditioning needed. Requires the weak
                    model to fail in the SAME direction as the strong one;
                    scripts/estimate_guidance_w.py tests that before sampling.

Both cost 2x NFE per step, which the caller must reflect in its cost accounting
or the efficiency numbers silently lie.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CFGVelocity(nn.Module):
    """Classifier-free guidance around one conditional model.

    The null token is the all-zero conditioning vector [0, 0]: zero value, zero
    mask. The mask channel is what makes that unambiguous -- a standardized
    value of 0 is the mean label, not the absence of one.
    """

    def __init__(self, model, w: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.w = float(w)

    def forward(self, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        if y is None:
            raise ValueError("CFGVelocity needs a conditioning vector")
        if self.w == 1.0:                      # plain conditional, one pass
            return self.model(x, t, y)
        if self.w == 0.0:                      # plain unconditional, one pass
            return self.model(x, t, torch.zeros_like(y))
        # one forward on a doubled batch: [null ; cond]
        x2 = torch.cat([x, x], 0)
        t2 = torch.cat([t, t], 0)
        y2 = torch.cat([torch.zeros_like(y), y], 0)
        v_un, v_co = self.model(x2, t2, y2).chunk(2, dim=0)
        return v_un + self.w * (v_co - v_un)


class AutoGuidedVelocity(nn.Module):
    """Autoguidance: extrapolate away from a deliberately degraded model.

    v0 must be degraded in the SAME way v1 is imperfect -- see
    scripts/estimate_guidance_w.py, which estimates E<e1, v1-v0> on held-out
    data and refuses the method when that is positive.
    """

    def __init__(self, v1, v0, w: float = 0.0) -> None:
        super().__init__()
        self.v1, self.v0, self.w = v1, v0, float(w)

    def forward(self, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        a = self.v1(x, t, y)
        if self.w == 0.0:
            return a
        return (1.0 + self.w) * a - self.w * self.v0(x, t, y)


def guided_nfe(base_nfe: int, w: float) -> int:
    """NFE after guidance: two field evaluations per step unless w collapses."""
    return base_nfe if w in (0.0, 1.0) else 2 * base_nfe
