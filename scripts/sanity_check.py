"""Correctness check for the flow matching implementation.

Trains on 2-D toy distributions where the right answer is known, and reports
quantitative metrics rather than asking you to eyeball a scatter plot:

  1. unconditional 8-gaussians  -> sliced Wasserstein-2 vs. held-out data
  2. conditional 8-gaussians    -> does conditioning on a mode put samples there
  3. paired transport           -> can it learn moons -> circle as data->data

Run:  .venv/bin/python scripts/sanity_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fm import (  # noqa: E402
    FlowMatching,
    PairedVectorDataset,
    TrainConfig,
    Trainer,
    VectorDataset,
    VelocityMLP,
    sample,
)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- toy datasets
def eight_gaussians(n: int, std: float = 0.15, radius: float = 2.0):
    """Returns (points, mode_index)."""
    idx = RNG.integers(0, 8, size=n)
    angles = idx * (2 * math.pi / 8)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    return (centers + std * RNG.standard_normal((n, 2))).astype(np.float32), idx


def two_moons(n: int, noise: float = 0.08):
    m = n // 2
    t = RNG.uniform(0, math.pi, m)
    a = np.stack([np.cos(t), np.sin(t)], 1)
    b = np.stack([1 - np.cos(t), -np.sin(t) + 0.3], 1)
    pts = np.concatenate([a, b], 0) * 2.0
    return (pts + noise * RNG.standard_normal(pts.shape)).astype(np.float32)


def ring(n: int, r: float = 2.0, noise: float = 0.1):
    t = RNG.uniform(0, 2 * math.pi, n)
    pts = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    return (pts + noise * RNG.standard_normal(pts.shape)).astype(np.float32)


# ------------------------------------------------------------------- metrics
def sliced_w2(a: np.ndarray, b: np.ndarray, n_proj: int = 256, seed: int = 0) -> float:
    """Sliced Wasserstein-2 distance: average 1-D W2 over random projections.

    Scale reference: two independent draws from the *same* distribution give a
    small nonzero value (the sampling floor). Compare against that, not zero.
    """
    g = np.random.default_rng(seed)
    d = a.shape[1]
    dirs = g.standard_normal((n_proj, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pa = np.sort(a @ dirs.T, axis=0)
    pb = np.sort(b @ dirs.T, axis=0)
    n = min(len(pa), len(pb))
    qs = np.linspace(0, 1, n)
    pa = np.stack([np.quantile(pa[:, j], qs) for j in range(n_proj)], 1)
    pb = np.stack([np.quantile(pb[:, j], qs) for j in range(n_proj)], 1)
    return float(np.sqrt(((pa - pb) ** 2).mean()))


def train(model, dataset, steps, out, has_cond=False, paired=False, bs=512):
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    cfg = TrainConfig(steps=steps, lr=2e-3, warmup_steps=200, log_every=max(steps // 4, 1),
                      val_every=10**9, ckpt_every=0, out_dir=out, device="cpu")
    Trainer(model, loader, cfg, has_cond=has_cond, paired=paired).fit()
    return model


def main() -> int:
    torch.manual_seed(0)
    failures = []

    # ---------------------------------------------------------------- test 1
    print("\n[1] unconditional 8-gaussians")
    x, _ = eight_gaussians(20_000)
    x_held, _ = eight_gaussians(4_000)
    model = FlowMatching(VelocityMLP(data_dim=2, width=256, depth=4))
    train(model, VectorDataset(x), 3_000, "runs/sanity_uncond")

    gen = sample(model.eval(), n=4_000, shape=(2,), n_steps=50, method="heun").cpu().numpy()
    floor = sliced_w2(x[:4_000], x_held)
    score = sliced_w2(gen, x_held)
    print(f"    sliced-W2 model-vs-data {score:.4f}   sampling floor {floor:.4f}")
    if score > 5 * floor + 0.05:
        failures.append(f"unconditional SW2 {score:.4f} too far above floor {floor:.4f}")

    # Euler vs Heun vs RK4: a correctly trained linear-path field should be
    # nearly solver-independent at enough steps.
    for method, nst in [("euler", 200), ("heun", 50), ("rk4", 25)]:
        g = sample(model, n=4_000, shape=(2,), n_steps=nst, method=method).cpu().numpy()
        print(f"    {method:>5s} ({nst:>3d} steps)  SW2 {sliced_w2(g, x_held):.4f}")

    # ---------------------------------------------------------------- test 2
    print("\n[2] conditional 8-gaussians (condition = one-hot mode)")
    x, idx = eight_gaussians(20_000)
    y = np.eye(8, dtype=np.float32)[idx]
    cmodel = FlowMatching(VelocityMLP(data_dim=2, cond_dim=8, width=256, depth=4))
    train(cmodel, VectorDataset(x, y), 3_000, "runs/sanity_cond", has_cond=True)

    radius, worst = 2.0, 0.0
    for mode in range(8):
        yy = torch.zeros(1000, 8)
        yy[:, mode] = 1.0
        g = sample(cmodel.eval(), n=1000, shape=(2,), y=yy, n_steps=50).cpu().numpy()
        ang = mode * (2 * math.pi / 8)
        center = np.array([radius * math.cos(ang), radius * math.sin(ang)])
        frac = float((np.linalg.norm(g - center, axis=1) < 0.6).mean())
        worst = max(worst, 1 - frac)
        print(f"    mode {mode}: {frac:6.1%} of samples within 0.6 of its center")
    if worst > 0.10:
        failures.append(f"conditional generation: worst mode only {1-worst:.1%} on target")

    # ---------------------------------------------------------------- test 3
    print("\n[3] paired transport, moons -> ring (data source, not noise)")
    src, tgt = two_moons(20_000), ring(20_000)
    tgt_held = ring(4_000)
    pmodel = FlowMatching(VelocityMLP(data_dim=2, width=256, depth=4))
    train(pmodel, PairedVectorDataset(src, tgt), 3_000, "runs/sanity_paired", paired=True)

    x0 = torch.from_numpy(two_moons(4_000))
    g = sample(pmodel.eval(), x0=x0, n_steps=50, method="heun").cpu().numpy()
    floor = sliced_w2(ring(4_000), tgt_held)
    score = sliced_w2(g, tgt_held)
    print(f"    sliced-W2 transported-vs-target {score:.4f}   floor {floor:.4f}")
    if score > 5 * floor + 0.05:
        failures.append(f"paired transport SW2 {score:.4f} too far above floor {floor:.4f}")

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 62)
    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
