"""Download MOSES, convert to SELFIES, tokenize into a fixed-length integer array.

MOSES is a standard molecular-generation benchmark (~1.9M ZINC-derived molecules)
with published baselines, which is exactly what you want when the goal is to
find out whether your flow matching implementation is correct. If your numbers
land near the published ones, it works; if not, it doesn't.

SELFIES rather than SMILES on purpose: every SELFIES string decodes to a valid
molecule by construction. That removes validity as a confound, so any failure
you see is a failure of the *latent flow matching*, not of string syntax. The
flip side, which matters when you read your results: validity near 100% proves
nothing here. Judge this track on FCD, novelty and uniqueness.

    .venv/bin/python scripts/prep_moses.py --n 100000
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"

PAD, BOS, EOS = "[pad]", "[bos]", "[eos]"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100_000, help="molecules to use (0 = all)")
    p.add_argument("--max-len", type=int, default=72, help="max SELFIES tokens; longer are dropped")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import selfies as sf

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv = data_dir / "moses.csv"
    if not csv.exists():
        print("downloading MOSES (~85 MB) ...")
        urllib.request.urlretrieve(URL, csv)

    import csv as csvmod

    smiles, splits = [], []
    with csv.open() as f:
        for row in csvmod.DictReader(f):
            smiles.append(row["SMILES"])
            splits.append(row.get("SPLIT", "train"))
    smiles, splits = np.array(smiles), np.array(splits)
    print(f"loaded {len(smiles):,} molecules  splits={dict(zip(*np.unique(splits, return_counts=True)))}")

    train_smiles = smiles[splits == "train"]
    if args.n and args.n < len(train_smiles):
        rng = np.random.default_rng(args.seed)
        train_smiles = train_smiles[rng.choice(len(train_smiles), args.n, replace=False)]
    print(f"using {len(train_smiles):,} training molecules")

    print("converting to SELFIES ...")
    encoded, kept_smiles = [], []
    for i, s in enumerate(train_smiles):
        if i and i % 25_000 == 0:
            print(f"  {i:,}/{len(train_smiles):,}")
        try:
            e = sf.encoder(s)
        except Exception:
            continue
        toks = list(sf.split_selfies(e))
        if len(toks) <= args.max_len - 1:  # room for EOS
            encoded.append(toks)
            kept_smiles.append(s)
    print(f"  kept {len(encoded):,} ({len(encoded)/len(train_smiles):.1%})")

    alphabet = sorted({t for toks in encoded for t in toks})
    itos = [PAD, BOS, EOS] + alphabet
    stoi = {t: i for i, t in enumerate(itos)}
    print(f"vocab: {len(itos)} tokens, max_len={args.max_len}")

    # tokens[i] is the target sequence: SELFIES tokens then EOS then PAD.
    n, L = len(encoded), args.max_len
    tokens = np.zeros((n, L), dtype=np.int16)
    for i, toks in enumerate(encoded):
        ids = [stoi[t] for t in toks] + [stoi[EOS]]
        tokens[i, : len(ids)] = ids

    np.save(data_dir / "moses_tokens.npy", tokens)
    (data_dir / "moses_vocab.json").write_text(
        json.dumps({"itos": itos, "max_len": L, "pad": 0, "bos": 1, "eos": 2}, indent=2)
    )
    (data_dir / "moses_train_smiles.txt").write_text("\n".join(kept_smiles))
    # Held-out reference set for FCD/novelty. Never train on this.
    test = smiles[splits == "test"] if (splits == "test").any() else smiles[:10_000]
    (data_dir / "moses_test_smiles.txt").write_text("\n".join(test[:25_000]))

    print(f"\nwrote {data_dir/'moses_tokens.npy'}  shape={tokens.shape}")
    print(f"      {data_dir/'moses_vocab.json'}")
    print(f"      {data_dir/'moses_test_smiles.txt'}  ({min(len(test),25_000):,} held-out)")
    print(f"\nnext:\n  .venv/bin/python scripts/train_latent_ae.py")


if __name__ == "__main__":
    main()
