"""Stage 2 of the latent track: flow matching in the VAE latent space.

The flow matching code here is the *same* code as the conformer track -- same
VelocityMLP, same CondOTPath, same trainer. Only the data changed. That is the
point of the exercise: once your data is a fixed-dimension vector, the modality
stops mattering.

Evaluation uses MOSES-style metrics so you can compare against published
numbers. Read them in this order:
  uniqueness -- catches mode collapse, the failure FM loss cannot see
  novelty    -- catches memorization of the training set
  validity   -- near 100% by SELFIES construction; it proves nothing here

    .venv/bin/python scripts/train_latent_fm.py --steps 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fm import (  # noqa: E402
    FlowMatching, Standardizer, TrainConfig, Trainer, VectorDataset, VelocityMLP, sample,
)
from fm.latent import SeqVAE  # noqa: E402


def moses_metrics(smiles: list[str], train_set: set[str], n_requested: int) -> dict:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    canon = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None:
            canon.append(Chem.MolToSmiles(m))
    uniq = set(canon)
    return {
        "validity": len(canon) / max(n_requested, 1),
        "uniqueness": len(uniq) / max(len(canon), 1),
        "novelty": len(uniq - train_set) / max(len(uniq), 1),
        "n_valid": len(canon),
        "n_unique": len(uniq),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--ae", default="runs/latent_ae/vae.pt")
    p.add_argument("--out", default="runs/latent_fm")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--n-samples", type=int, default=5_000)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                       if args.device == "auto" else args.device)
    data_dir, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    z = torch.from_numpy(np.load(data_dir / "moses_latents.npy"))
    D = z.shape[1]
    scaler = Standardizer().fit(z)
    z = scaler.transform(z)
    print(f"latents {tuple(z.shape)}  device={dev}")

    ds = VectorDataset(z)
    n_val = 5_000
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))
    model = FlowMatching(VelocityMLP(D, width=args.width, depth=args.depth))
    print(f"velocity field: {sum(q.numel() for q in model.parameters())/1e6:.2f}M params")

    cfg = TrainConfig(steps=args.steps, lr=args.lr, out_dir=str(out), device=str(dev))
    trainer = Trainer(
        model,
        DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True),
        cfg,
        DataLoader(va, batch_size=512),
    )
    trainer.fit()

    ema = FlowMatching(VelocityMLP(D, width=args.width, depth=args.depth))
    ema.load_state_dict(trainer.ema.state_dict())
    ema.to(dev).eval()

    gen_z = sample(ema, n=args.n_samples, shape=(D,), n_steps=args.ode_steps,
                   method="heun", device=dev)
    gen_z = scaler.inverse(gen_z.cpu()).to(dev)

    # Decode the generated latents back to molecules.
    ck = torch.load(args.ae, map_location=dev, weights_only=False)
    a, vocab = ck["args"], ck["vocab"]
    vae = SeqVAE(len(vocab["itos"]), a["latent_dim"], a["emb_dim"], a["hidden"],
                 beta=a["beta"]).to(dev)
    vae.load_state_dict(ck["model"])
    vae.eval()

    import selfies as sf
    itos = vocab["itos"]
    toks = vae.generate(gen_z, vocab["max_len"], vocab["bos"], vocab["eos"]).cpu().numpy()
    smiles = []
    for row in toks:
        s = "".join(itos[t] for t in row if t > 2)  # drop pad/bos/eos
        try:
            smiles.append(sf.decoder(s))
        except Exception:
            smiles.append("")

    # Canonicalize the training set with the same parser used on the samples,
    # or novelty is measured against differently-written versions of the same
    # molecules and comes out spuriously high.
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    train_set = set()
    for s_ in (data_dir / "moses_train_smiles.txt").read_text().split():
        m_ = Chem.MolFromSmiles(s_)
        if m_ is not None:
            train_set.add(Chem.MolToSmiles(m_))

    # Autoencoder ceiling: decode REAL latents through the same decoder.
    # This is the single most useful diagnostic in this track, because low
    # uniqueness has two completely different causes that look identical in the
    # metrics table. If the ceiling is high and the model is low, flow matching
    # mode-collapsed. If the ceiling is ALSO low, the decoder collapsed and no
    # amount of flow matching work will help -- go fix the autoencoder.
    real_z = torch.from_numpy(np.load(data_dir / "moses_latents.npy"))
    real_z = real_z[torch.randperm(len(real_z))[: args.n_samples]].to(dev)
    real_toks = vae.generate(real_z, vocab["max_len"], vocab["bos"], vocab["eos"]).cpu().numpy()
    ceil_smiles = []
    for row in real_toks:
        try:
            ceil_smiles.append(sf.decoder("".join(itos[t] for t in row if t > 2)))
        except Exception:
            ceil_smiles.append("")
    m = moses_metrics(smiles, train_set, args.n_samples)
    ceil = moses_metrics(ceil_smiles, train_set, args.n_samples)
    m["ae_ceiling_uniqueness"] = ceil["uniqueness"]

    print("\n" + "=" * 62)
    print(f"{'metric':<22s} {'model':>10s} {'AE ceiling':>12s}   {'reads on':>12s}")
    print("-" * 62)
    notes = {"validity": "nothing", "uniqueness": "collapse", "novelty": "memorization"}
    for k in ("validity", "uniqueness", "novelty"):
        print(f"{k:<22s} {m[k]:>10.4f} {ceil[k]:>12.4f}   {notes[k]:>12s}")
    print(f"{'n_unique':<22s} {m['n_unique']:>10d} {ceil['n_unique']:>12d}")

    print()
    if ceil["uniqueness"] < 0.5:
        print("DIAGNOSIS: the AE ceiling is low too, so the DECODER collapsed --")
        print("  real latents decode to near-identical junk. Flow matching is not")
        print("  the problem. Train the autoencoder longer / on more data first.")
    elif m["uniqueness"] < 0.5 * ceil["uniqueness"]:
        print("DIAGNOSIS: the AE decodes fine but the model does not, so FLOW")
        print("  MATCHING mode-collapsed. Train longer, or check standardization.")
    else:
        print("Model is tracking the autoencoder ceiling -- the FM stage is healthy.")
    print("\nNote how validity and novelty can BOTH read 1.0 while the model is")
    print("entirely broken. Uniqueness is the metric that catches it.")
    (out / "samples.txt").write_text("\n".join(s for s in smiles if s))
    json.dump(m, (out / "metrics.json").open("w"), indent=2)
    print(f"\nwrote {out/'samples.txt'}")


if __name__ == "__main__":
    main()
