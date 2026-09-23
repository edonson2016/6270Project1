"""Enumerate cation.anion combinations from TRAIN ions, for decoder training only.

Motivation (docs §6.9): at d=16 flow matching already closes 98.2% of the
prior->ceiling gap, so the FM-vs-DDPM comparison at the best operating point has
almost no headroom left. The binding constraint is the decoder ceiling, and the
decoder is trained on only 3,447 ion pairs.

The observed pairs are 0.57% of the cation x anion grid. Every unobserved
combination still pairs two REAL ions from measured ILs, so it is valid
supervision for the decoder's actual job (z -> SMILES) even though nobody has
reported that particular salt.

WHAT THIS IS NOT FOR. The flow must never train on these. The observed 4,790 are
not a random slice of the grid -- they are what chemists actually made, which
encodes synthesizability and stability, and that structure is the thing worth
modelling. Train the flow on enumerated pairs and you teach it the uniform
product distribution instead.

SPLIT SAFETY. Enumerating from all 2,253 cations would leak test_ion cations into
decoder training and destroy the held-out evaluation. This enumerates from TRAIN
ions only and asserts no held-out pair or test_ion cation appears.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm.il_eval import make_splits, ion_sets

DATA = Path("data")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    ils = json.loads((DATA / "il_pairs_v2.json").read_text())
    sp = make_splits(ils, seed=a.split_seed)
    trP, trC, trA = ion_sets(ils, sp["train"])

    # everything the decoder must not see, as canonical "cation.anion" strings
    held = set()
    for k in ("valid", "test", "test_ion"):
        held |= ion_sets(ils, sp[k])[0]
    ood = json.loads((DATA / "il_ood_pairs.json").read_text())
    held |= {q["pair"] for q in ood}
    heldC = ion_sets(ils, sp["test_ion"])[1]

    C, A = sorted(trC), sorted(trA)
    rng = np.random.default_rng(a.seed)
    grid = len(C) * len(A)
    print(f"train ions: {len(C)} cations x {len(A)} anions = {grid:,} combinations")
    print(f"observed in train: {len(trP):,}   held-out pairs to exclude: {len(held):,}")

    # sample distinct grid cells without materializing 568k strings
    want = min(a.n, grid)
    picked, pairs = set(), []
    while len(pairs) < want:
        for idx in rng.choice(grid, size=min(want * 2, grid), replace=False):
            if len(pairs) >= want: break
            if idx in picked: continue
            picked.add(int(idx))
            s = f"{C[idx // len(A)]}.{A[idx % len(A)]}"
            if s in held: continue          # never emit a held-out pair
            pairs.append(s)
        if len(picked) >= grid: break

    # the observed train pairs belong in decoder training too
    pairs = sorted(set(pairs) | trP)
    rng.shuffle(pairs)

    # hard guarantees
    assert not (set(pairs) & held), "held-out pair leaked into enumeration"
    leak = {s.split(".")[0] for s in pairs} & heldC
    assert not leak, f"test_ion cation leaked: {len(leak)}"

    (DATA / "il_smi_enum.txt").write_text("\n".join(pairs))
    print(f"wrote {len(pairs):,} pairs -> {DATA/'il_smi_enum.txt'}")
    print(f"  observed train pairs included : {len(trP):,}")
    print(f"  novel combinations added      : {len(pairs) - len(trP):,}")
    print(f"  decoder supervision vs before : {len(pairs)/len(sp['train']):.0f}x")
    print("  asserts passed: no held-out pair, no test_ion cation")


if __name__ == "__main__":
    main()
