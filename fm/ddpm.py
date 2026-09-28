"""DDPM — the discrete-time diffusion baseline, for comparison against flow matching.

Deliberately built to be swappable with `FlowMatching`: same `loss(x1, y, x0)`
signature, same network, same Trainer, same EMA. Only the objective and the
sampler differ, so a comparison between the two isolates exactly that.

    train:   x0 ~ data,  i ~ U{0..T-1},  eps ~ N(0, I)
             x_i  = sqrt(abar_i) * x0 + sqrt(1 - abar_i) * eps
             loss = || eps_theta(x_i, i) - eps ||^2

    sample:  x_T ~ N(0, I), then walk i = T-1 .. 0 (ancestral or DDIM)

TIME RUNS THE OTHER WAY FROM THE REST OF THIS PACKAGE. In `fm/paths.py`, t=0 is
noise and t=1 is data. Here index i=0 is *data* and i=T-1 is *noise*, which is
the standard diffusion convention and the usual source of sign confusion when
the two are read side by side. The network sees `(i+1)/T` in (0,1], so a larger
input still means "more noise" and `SinusoidalTimeEmbedding` -- which multiplies
its argument by 1000 -- receives exactly the integer timestep when T=1000, the
standard DDPM embedding.

The network is the same `VelocityMLP`, reinterpreted: it predicts eps rather
than a velocity. Identical parameter count and identical FLOPs per evaluation,
which is what makes the efficiency comparison fair.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> Tensor:
    """Ho et al. 2020. Standard, and tuned there for 32x32 images at T=1000.

    On low-dimensional data it destroys information relatively slowly at the
    start, which is the usual argument for the cosine schedule below. Kept as
    the default anyway, because "standard DDPM" is what it means.
    """
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


def cosine_beta_schedule(T: int, s: float = 0.008, max_beta: float = 0.999) -> Tensor:
    """Nichol & Dhariwal 2021, defined through abar rather than beta."""
    i = torch.arange(T + 1, dtype=torch.float32) / T
    abar = torch.cos((i + s) / (1.0 + s) * math.pi / 2.0) ** 2
    abar = abar / abar[0]
    return torch.clip(1.0 - abar[1:] / abar[:-1], 0.0, max_beta)


class DDPM(nn.Module):
    """Epsilon-prediction DDPM over flat vectors.

    Args:
        net: module with signature (x, t, y) -> output shaped like x. The same
            `VelocityMLP` used for flow matching; here its output is eps.
        T: number of diffusion steps.
        schedule: "linear" (Ho et al.) or "cosine" (Nichol & Dhariwal).
    """

    def __init__(self, net: nn.Module, T: int = 1000, schedule: str = "linear") -> None:
        super().__init__()
        self.net = net
        self.T = int(T)
        self.schedule = schedule
        betas = {"linear": linear_beta_schedule, "cosine": cosine_beta_schedule}[schedule](self.T)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        abar_prev = torch.cat([torch.ones(1), abar[:-1]])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)
        self.register_buffer("abar_prev", abar_prev)
        # posterior variance of q(x_{i-1} | x_i, x_0); index 0 is 0, so it is
        # clamped when used as a log or a sqrt
        self.register_buffer("post_var", betas * (1.0 - abar_prev) / (1.0 - abar).clamp_min(1e-20))

    # the network's time input: (i+1)/T in (0,1], increasing with noise
    def _tnorm(self, i: Tensor) -> Tensor:
        return (i.float() + 1.0) / self.T

    def forward(self, x: Tensor, i: Tensor, y: Tensor | None = None) -> Tensor:
        """Predict eps at integer timestep(s) i."""
        return self.net(x, self._tnorm(i), y)

    def q_sample(self, x0: Tensor, i: Tensor, eps: Tensor) -> Tensor:
        a = self.abar[i].view(-1, *([1] * (x0.dim() - 1)))
        return a.sqrt() * x0 + (1.0 - a).sqrt() * eps

    def loss(self, x1: Tensor, y: Tensor | None = None, x0: Tensor | None = None,
             reduction: str = "mean") -> Tensor:
        """Simplified DDPM objective (L_simple): MSE on the noise.

        `x1` is the data, named to match `FlowMatching.loss` so the same Trainer
        drives both. `x0` has no meaning here -- vanilla DDPM always starts from
        Gaussian noise, so data-to-data transport is not defined.
        """
        if x0 is not None:
            raise NotImplementedError("DDPM has no paired-source mode; use FlowMatching")
        i = torch.randint(0, self.T, (x1.shape[0],), device=x1.device)
        eps = torch.randn_like(x1)
        x_i = self.q_sample(x1, i, eps)
        pred = self.forward(x_i, i, y)
        per_sample = (pred - eps).pow(2).flatten(1).mean(dim=1)
        if reduction == "none":
            return per_sample
        if reduction == "sum":
            return per_sample.sum()
        return per_sample.mean()

    # ------------------------------------------------------------- samplers
    @torch.no_grad()
    def sample(self, n: int, shape: tuple[int, ...], y: Tensor | None = None,
               device=None, sampler: str = "ancestral", n_steps: int | None = None,
               eta: float = 0.0, clip_x0: float | None = None,
               guide_w: float = 1.0) -> Tensor:
        """Draw samples.

        sampler="ancestral": the original DDPM reverse chain, T network
            evaluations -- 1000 by default, which is the cost that matters.
        sampler="ddim": Song et al. 2021, a deterministic (eta=0) sub-sequence
            of `n_steps` timesteps. This is the setting that puts diffusion on
            the same evaluation budget as an ODE solver.

        Both are written in the predict-x0 form, which is algebraically identical
        to the eps form but exposes the one knob that matters numerically:

        `guide_w` applies classifier-free guidance to the eps prediction:
        eps_un + w * (eps_cond - eps_un), evaluated as one doubled batch. w = 1 is
        plain conditional sampling and costs T evaluations; w != 1 costs 2T, which
        n_function_evals does NOT know about -- the caller must double it.

        `clip_x0` clamps the predicted x0 to +/- that many units before taking
        the posterior mean (Ho et al. do this for images with a [-1,1] range).
        It is off by default because plain DDPM is the baseline being measured.
        It is close to mandatory with the cosine schedule, where abar_T ~ 2e-9
        means x0_pred divides by ~5e-5 and any eps error is amplified by 2e4.
        On standardized data a clip of 3-5 is the natural range.
        """
        device = device or self.betas.device
        x = torch.randn(n, *shape, device=device)
        if sampler == "ancestral":
            steps = list(range(self.T - 1, -1, -1))
        elif sampler == "ddim":
            k = n_steps or 50
            steps = list(reversed(torch.linspace(0, self.T - 1, k).round().long().tolist()))
        else:
            raise ValueError(f"unknown sampler {sampler!r}")

        for n_i, i in enumerate(steps):
            it = torch.full((n,), i, device=device, dtype=torch.long)
            if y is not None and guide_w != 1.0:
                e_un, e_co = self.forward(torch.cat([x, x]), torch.cat([it, it]),
                                          torch.cat([torch.zeros_like(y), y])).chunk(2, 0)
                eps = e_un + guide_w * (e_co - e_un)
            else:
                eps = self.forward(x, it, y)
            a_i = self.abar[i]
            x0_pred = (x - (1.0 - a_i).clamp_min(1e-20).sqrt() * eps) / a_i.clamp_min(1e-20).sqrt()
            if clip_x0 is not None:
                x0_pred = x0_pred.clamp(-clip_x0, clip_x0)

            if sampler == "ancestral":
                a_prev = self.abar_prev[i]
                # posterior mean of q(x_{i-1} | x_i, x_0)
                c0 = a_prev.sqrt() * self.betas[i] / (1.0 - a_i).clamp_min(1e-20)
                ci = self.alphas[i].sqrt() * (1.0 - a_prev) / (1.0 - a_i).clamp_min(1e-20)
                mean = c0 * x0_pred + ci * x
                x = mean if i == 0 else mean + self.post_var[i].clamp_min(0).sqrt() * torch.randn_like(x)
            else:
                i_prev = steps[n_i + 1] if n_i + 1 < len(steps) else -1
                a_prev = self.abar[i_prev] if i_prev >= 0 else torch.ones((), device=device)
                sigma = eta * (((1 - a_prev) / (1 - a_i)).clamp_min(0)
                               * (1 - a_i / a_prev).clamp_min(0)).sqrt()
                # re-derive eps from the (possibly clipped) x0 so the two agree
                eps_use = (x - a_i.sqrt() * x0_pred) / (1.0 - a_i).clamp_min(1e-20).sqrt()
                dir_xt = (1.0 - a_prev - sigma ** 2).clamp_min(0).sqrt() * eps_use
                x = a_prev.sqrt() * x0_pred + dir_xt
                if eta > 0 and i_prev >= 0:
                    x = x + sigma * torch.randn_like(x)
        return x


def n_function_evals(sampler: str, T: int, n_steps: int | None = None,
                     ode_method: str = "heun") -> int:
    """Network evaluations to produce one sample -- the unit of sampling cost.

    Flow matching pays (solver stages x steps); DDPM ancestral pays T.
    """
    if sampler == "ancestral":
        return T
    if sampler == "ddim":
        return n_steps or 50
    return {"euler": 1, "heun": 2, "rk4": 4}[ode_method] * (n_steps or 50)
