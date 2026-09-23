"""Dataset plumbing for fixed-dimension vector data.

Everything here assumes samples are flat vectors of a fixed dimension D. That
covers spectra, expression profiles, latent codes, descriptor vectors, and the
internal coordinates of a fixed-composition molecular system.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class Standardizer:
    """Per-feature zero-mean/unit-variance scaling.

    Flow matching from a standard Gaussian source works far better when the
    target data is on a comparable scale. This is the single highest-leverage
    preprocessing step; skipping it is the most common reason a first FM run
    looks broken.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.mean: Tensor | None = None
        self.std: Tensor | None = None
        self.eps = eps

    def fit(self, x: Tensor) -> "Standardizer":
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True).clamp_min(self.eps)
        return self

    def transform(self, x: Tensor) -> Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse(self, x: Tensor) -> Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    def load_state_dict(self, sd) -> "Standardizer":
        self.mean, self.std, self.eps = sd["mean"], sd["std"], sd["eps"]
        return self


class VectorDataset(Dataset):
    """Unpaired data: generate x1 from noise, optionally conditioned on y."""

    def __init__(self, x1: np.ndarray | Tensor, y: np.ndarray | Tensor | None = None) -> None:
        self.x1 = torch.as_tensor(np.asarray(x1), dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(np.asarray(y), dtype=torch.float32)
        if self.y is not None and len(self.y) != len(self.x1):
            raise ValueError("x1 and y must have the same length")

    def __len__(self) -> int:
        return len(self.x1)

    def __getitem__(self, i: int):
        if self.y is None:
            return self.x1[i]
        return self.x1[i], self.y[i]


class PairedVectorDataset(Dataset):
    """Transport between two data distributions.

    Use this when the scientific question is "given a system in state A, what
    does the distribution over state B look like" -- source is real data rather
    than noise. The coupling is whatever pairing you supply; if you have no
    natural pairing, an independent (randomly re-shuffled each epoch) coupling
    is the standard choice.
    """

    def __init__(
        self,
        x0: np.ndarray | Tensor,
        x1: np.ndarray | Tensor,
        y: np.ndarray | Tensor | None = None,
        resample_source: bool = True,
    ) -> None:
        self.x0 = torch.as_tensor(np.asarray(x0), dtype=torch.float32)
        self.x1 = torch.as_tensor(np.asarray(x1), dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(np.asarray(y), dtype=torch.float32)
        self.resample_source = resample_source
        if self.x0.shape[1:] != self.x1.shape[1:]:
            raise ValueError("x0 and x1 must have the same feature shape")

    def __len__(self) -> int:
        return len(self.x1)

    def __getitem__(self, i: int):
        j = torch.randint(len(self.x0), (1,)).item() if self.resample_source else i
        out = [self.x0[j], self.x1[i]]
        if self.y is not None:
            out.append(self.y[i])
        return tuple(out)


def unpack_batch(batch, has_cond: bool, paired: bool):
    """Normalize a dataloader batch into (x1, y, x0)."""
    if paired:
        if has_cond:
            x0, x1, y = batch
            return x1, y, x0
        x0, x1 = batch
        return x1, None, x0
    if has_cond:
        x1, y = batch
        return x1, y, None
    return batch, None, None
