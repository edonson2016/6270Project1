"""ODE integration for sampling.

Sampling = solve  dx/dt = v_theta(x, t, y)  from t=0 (source) to t=1 (data).

Note on what t means: this is the *path* time of the generative process, not
physical time. The intermediate states x_t are points on an interpolation
between noise and data. They are not physical intermediates, transition states,
or folding intermediates, however suggestive they look when you animate them.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


@torch.no_grad()
def integrate(
    v: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    n_steps: int = 100,
    method: str = "heun",
    t0: float = 0.0,
    t1: float = 1.0,
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Fixed-step ODE solve.

    Args:
        v: callable (x, t_scalar_batch) -> velocity. Bind conditioning with a
           lambda or functools.partial before passing it in.
        x0: initial state, shape (B, ...).
        n_steps: number of steps. Euler needs ~100+; heun/rk4 are usually fine
           with 20-50 on a well-trained linear-path model.
        method: "euler", "heun" (2nd order), or "rk4" (4th order).

    Returns:
        x1, or (x1, trajectory) with trajectory of shape (n_steps + 1, B, ...).
    """
    method = method.lower()
    if method not in {"euler", "heun", "rk4"}:
        raise ValueError(f"unknown method {method!r}")

    x = x0
    dt = (t1 - t0) / n_steps
    traj = [x0] if return_trajectory else None

    def tvec(t: float) -> Tensor:
        return torch.full((x.shape[0],), t, device=x0.device, dtype=x0.dtype)

    for i in range(n_steps):
        t = t0 + i * dt
        if method == "euler":
            x = x + dt * v(x, tvec(t))
        elif method == "heun":
            k1 = v(x, tvec(t))
            x_pred = x + dt * k1
            k2 = v(x_pred, tvec(t + dt))
            x = x + 0.5 * dt * (k1 + k2)
        else:  # rk4
            k1 = v(x, tvec(t))
            k2 = v(x + 0.5 * dt * k1, tvec(t + 0.5 * dt))
            k3 = v(x + 0.5 * dt * k2, tvec(t + 0.5 * dt))
            k4 = v(x + dt * k3, tvec(t + dt))
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if return_trajectory:
            traj.append(x)

    if return_trajectory:
        return x, torch.stack(traj, dim=0)
    return x


@torch.no_grad()
def sample(
    model,
    n: int | None = None,
    shape: tuple[int, ...] | None = None,
    y: Tensor | None = None,
    x0: Tensor | None = None,
    n_steps: int = 50,
    method: str = "heun",
    device=None,
    return_trajectory: bool = False,
):
    """Draw samples from a trained FlowMatching model.

    Either pass x0 directly, or pass n and shape to draw x0 ~ N(0, I).
    """
    model.eval()
    if x0 is None:
        if n is None or shape is None:
            raise ValueError("pass either x0, or both n and shape")
        device = device or next(model.parameters()).device
        x0 = torch.randn(n, *shape, device=device)
    if y is not None:
        y = y.to(x0.device)

    def v(x: Tensor, t: Tensor) -> Tensor:
        return model(x, t, y)

    return integrate(
        v, x0, n_steps=n_steps, method=method, return_trajectory=return_trajectory
    )
