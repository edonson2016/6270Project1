"""Train a flow matching model on your own fixed-dimension vector data.

Expects .npy arrays of shape (N, D). Conditioning and paired-source are optional.

Unconditional generation:
    .venv/bin/python scripts/train_vectors.py --x1 data/x1.npy --out runs/mine

Conditional generation:
    .venv/bin/python scripts/train_vectors.py --x1 data/x1.npy --y data/y.npy --out runs/mine

Transport between two data distributions (source is data, not noise):
    .venv/bin/python scripts/train_vectors.py --x0 data/ctrl.npy --x1 data/pert.npy --out runs/mine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fm import (  # noqa: E402
    FlowMatching,
    PairedVectorDataset,
    Standardizer,
    TrainConfig,
    Trainer,
    VectorDataset,
    VelocityMLP,
    sample,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--x1", required=True, help="target data, .npy of shape (N, D)")
    p.add_argument("--x0", default=None, help="optional source data, .npy of shape (M, D)")
    p.add_argument("--y", default=None, help="optional conditioning, .npy of shape (N, C)")
    p.add_argument("--out", default="runs/vectors")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--n-samples", type=int, default=2_000, help="samples to draw after training")
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    x1 = torch.as_tensor(np.load(args.x1), dtype=torch.float32)
    if x1.dim() != 2:
        raise SystemExit(f"--x1 must be 2-D (N, D); got {tuple(x1.shape)}")
    D = x1.shape[1]

    # Standardize the target. Do this. See fm/data.py for why.
    scaler = Standardizer().fit(x1)
    x1 = scaler.transform(x1)
    torch.save(scaler.state_dict(), out / "scaler.pt")

    paired = args.x0 is not None
    x0 = None
    if paired:
        x0 = torch.as_tensor(np.load(args.x0), dtype=torch.float32)
        if x0.shape[1] != D:
            raise SystemExit(f"--x0 dim {x0.shape[1]} != --x1 dim {D}")
        # Same scaler for both ends, so the transport stays in one coordinate system.
        x0 = scaler.transform(x0)

    y, cond_dim = None, 0
    if args.y:
        y = torch.as_tensor(np.load(args.y), dtype=torch.float32)
        if y.dim() == 1:
            y = y[:, None]
        if len(y) != len(x1):
            raise SystemExit(f"--y has {len(y)} rows, --x1 has {len(x1)}")
        cond_dim = y.shape[1]

    ds = PairedVectorDataset(x0, x1, y) if paired else VectorDataset(x1, y)
    n_val = int(len(ds) * args.val_frac)
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if n_val else None

    print(f"D={D}  N={len(ds)}  cond_dim={cond_dim}  paired={paired}  val={n_val}")

    model = FlowMatching(
        VelocityMLP(D, cond_dim=cond_dim, width=args.width, depth=args.depth, dropout=args.dropout)
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"velocity field: {n_params/1e6:.2f}M params")

    cfg = TrainConfig(
        steps=args.steps, lr=args.lr, out_dir=str(out), device=args.device, seed=args.seed
    )
    trainer = Trainer(model, train_loader, cfg, val_loader, has_cond=cond_dim > 0, paired=paired)
    trainer.fit()

    # Sample from the EMA weights -- consistently better than the raw weights.
    ema_model = FlowMatching(
        VelocityMLP(D, cond_dim=cond_dim, width=args.width, depth=args.depth, dropout=args.dropout)
    )
    ema_model.load_state_dict(trainer.ema.state_dict())
    ema_model.to(trainer.device).eval()

    n = args.n_samples
    kw = {"n_steps": args.ode_steps, "method": "heun"}
    if paired:
        idx = torch.randint(len(x0), (n,))
        gen = sample(ema_model, x0=x0[idx].to(trainer.device), y=(y[idx] if y is not None else None), **kw)
    else:
        yy = y[torch.randint(len(y), (n,))] if y is not None else None
        gen = sample(ema_model, n=n, shape=(D,), y=yy, device=trainer.device, **kw)

    gen = scaler.inverse(gen.cpu())
    np.save(out / "samples.npy", gen.numpy())
    print(f"wrote {n} samples -> {out/'samples.npy'}")
    print(f"  data  mean/std: {scaler.mean.mean():.4f} / {scaler.std.mean():.4f}")
    print(f"  model mean/std: {gen.mean():.4f} / {gen.std():.4f}")


if __name__ == "__main__":
    main()
