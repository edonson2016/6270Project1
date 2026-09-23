"""Training loop."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import unpack_batch
from .ema import EMA


@dataclass
class TrainConfig:
    steps: int = 20_000
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 200
    val_every: int = 1_000
    ckpt_every: int = 5_000
    out_dir: str = "runs/default"
    device: str = "auto"
    seed: int = 0


def _lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to 5% of peak."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def _infinite(loader: DataLoader):
    while True:
        yield from loader


class Trainer:
    """Trains a FlowMatching model on a DataLoader of vectors.

    Args:
        model: a FlowMatching instance.
        train_loader / val_loader: yield either x1, (x1, y), (x0, x1), or
            (x0, x1, y) depending on `has_cond` and `paired`.
        has_cond: whether batches carry a conditioning tensor.
        paired: whether batches carry an explicit source x0.
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        cfg: TrainConfig,
        val_loader: DataLoader | None = None,
        has_cond: bool = False,
        paired: bool = False,
    ) -> None:
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.has_cond = has_cond
        self.paired = paired

        self.opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999)
        )
        self.ema = EMA(model, cfg.ema_decay)

        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        self._log_path = self.out / "log.csv"
        with self._log_path.open("w", newline="") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss", "lr", "secs"])

    def _to_device(self, *tensors):
        return [None if t is None else t.to(self.device, non_blocking=True) for t in tensors]

    def _batch_loss(self, batch) -> torch.Tensor:
        x1, y, x0 = unpack_batch(batch, self.has_cond, self.paired)
        x1, y, x0 = self._to_device(x1, y, x0)
        return self.model.loss(x1, y=y, x0=x0)

    @torch.no_grad()
    def validate(self, max_batches: int = 50) -> float:
        """Validation FM loss.

        Caveat: this is a noisy quantity even at convergence, because the target
        u_t = x_1 - x_0 is not a deterministic function of (x_t, t). It is useful
        for spotting divergence and overfitting, not for judging sample quality.
        Always pair it with a domain metric computed on actual samples.
        """
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        total, n = 0.0, 0
        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            total += self._batch_loss(batch).item()
            n += 1
        self.model.train()
        return total / max(n, 1)

    def save(self, name: str) -> Path:
        path = self.out / name
        torch.save(
            {"model": self.model.state_dict(), "ema": self.ema.state_dict(), "cfg": asdict(self.cfg)},
            path,
        )
        return path

    def fit(self) -> None:
        cfg = self.cfg
        self.model.train()
        stream = _infinite(self.train_loader)
        start = time.time()
        running, n_run = 0.0, 0
        best_val = float("inf")

        for step in range(cfg.steps):
            lr = _lr_at(step, cfg)
            for g in self.opt.param_groups:
                g["lr"] = lr

            loss = self._batch_loss(next(stream))
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()
            self.ema.update(self.model)

            running += loss.item()
            n_run += 1

            if (step + 1) % cfg.log_every == 0:
                train_loss = running / n_run
                running, n_run = 0.0, 0
                val_loss = self.validate() if (step + 1) % cfg.val_every == 0 else float("nan")
                secs = time.time() - start
                with self._log_path.open("a", newline="") as f:
                    csv.writer(f).writerow(
                        [step + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.3e}", f"{secs:.1f}"]
                    )
                msg = f"step {step+1:>7d}/{cfg.steps}  train {train_loss:.5f}"
                if not math.isnan(val_loss):
                    msg += f"  val {val_loss:.5f}"
                    if val_loss < best_val:
                        best_val = val_loss
                        self.save("best.pt")
                        msg += "  *"
                print(f"{msg}  lr {lr:.2e}  {secs:.0f}s", flush=True)

            if cfg.ckpt_every and (step + 1) % cfg.ckpt_every == 0:
                self.save("last.pt")

        self.save("last.pt")
        print(f"done in {time.time() - start:.0f}s -> {self.out}", flush=True)
