"""Stage 1 of the latent track: train the sequence VAE and encode the dataset.

Flow matching never sees a molecule here. It will only ever see the latent
vectors this script produces. So the quality ceiling of the whole track is set
right here -- check the reported reconstruction accuracy before moving on. If
the autoencoder cannot round-trip its own training data, no amount of flow
matching will save you.

    .venv/bin/python scripts/train_latent_ae.py --epochs 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fm.latent import SeqVAE  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="runs/latent_ae")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--beta", type=float, default=3e-3, help="KL weight; the key knob")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n", type=int, default=0, help="subsample molecules (0 = all)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                       if args.device == "auto" else args.device)
    data_dir, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vocab = json.loads((data_dir / "moses_vocab.json").read_text())
    tokens = torch.from_numpy(np.load(data_dir / "moses_tokens.npy").astype(np.int64))
    if args.n and args.n < len(tokens):
        g = torch.Generator().manual_seed(0)
        tokens = tokens[torch.randperm(len(tokens), generator=g)[: args.n]]
    V, BOS = len(vocab["itos"]), vocab["bos"]
    print(f"{len(tokens):,} molecules  vocab={V}  max_len={tokens.shape[1]}  device={dev}")

    # decoder input is the target shifted right and prefixed with BOS
    tokens_in = torch.cat([torch.full((len(tokens), 1), BOS, dtype=torch.long), tokens[:, :-1]], 1)

    n_val = min(5_000, len(tokens) // 10)
    ds = TensorDataset(tokens, tokens_in)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    model = SeqVAE(V, args.latent_dim, args.emb_dim, args.hidden, beta=args.beta).to(dev)
    print(f"VAE: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params, "
          f"latent_dim={args.latent_dim}, beta={args.beta}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(train_loader))

    start = time.time()
    for ep in range(args.epochs):
        model.train()
        agg = {"recon": 0.0, "kl": 0.0, "token_acc": 0.0, "n": 0}
        for tgt, inp in train_loader:
            tgt, inp = tgt.to(dev), inp.to(dev)
            o = model(tgt, inp, tgt)
            opt.zero_grad(set_to_none=True)
            o["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k in ("recon", "kl", "token_acc"):
                agg[k] += float(o[k].detach())
            agg["n"] += 1
        n = agg["n"]

        model.eval()
        exact, total = 0, 0
        with torch.no_grad():
            for tgt, inp in val_loader:
                tgt, inp = tgt.to(dev), inp.to(dev)
                mu, _ = model.encode(tgt)
                pred = model.decode(mu, inp).argmax(-1)
                mask = tgt != 0
                exact += int(((pred == tgt) | ~mask).all(1).sum())
                total += len(tgt)
        print(f"epoch {ep+1:>2d}/{args.epochs}  recon {agg['recon']/n:.4f}  kl {agg['kl']/n:.3f}  "
              f"tok-acc {agg['token_acc']/n:.4f}  val exact-seq {exact/total:.4f}  "
              f"{time.time()-start:.0f}s", flush=True)

    torch.save({"model": model.state_dict(), "args": vars(args), "vocab": vocab}, out / "vae.pt")

    # Encode everything to latents -- this is what flow matching will train on.
    model.eval()
    lat = []
    with torch.no_grad():
        for i in range(0, len(tokens), 1024):
            mu, _ = model.encode(tokens[i:i+1024].to(dev))
            lat.append(mu.cpu())
    lat = torch.cat(lat).numpy().astype(np.float32)
    np.save(Path(args.data_dir) / "moses_latents.npy", lat)

    print(f"\nlatents: {lat.shape}  per-dim std range [{lat.std(0).min():.3f}, {lat.std(0).max():.3f}]")
    print(f"wrote {Path(args.data_dir)/'moses_latents.npy'}")
    print(f"\nnext:\n  .venv/bin/python scripts/train_latent_fm.py")


if __name__ == "__main__":
    main()
