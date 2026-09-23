"""Score generated conformers against the reference ensemble.

No quantum chemistry required. These metrics answer the question that the flow
matching loss cannot: did the model learn chemistry, or did it just learn to put
atoms in roughly the right region of space?

    .venv/bin/python scripts/eval_conformers.py \
        --samples runs/ethanol/samples.npy --meta data/ethanol_aligned_meta.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fm.featurize import (  # noqa: E402
    PairwiseDistance, bond_lengths, embedding_quality, geometry_report,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True, help=".npy of generated feature vectors")
    p.add_argument("--meta", required=True, help="the *_meta.pkl from prep_rmd17.py")
    args = p.parse_args()

    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    feat, z, bonds = meta["featurizer"], meta["z"], meta["bonds"]
    ref = meta["coords_sample"]
    ref = ref - ref.mean(axis=1, keepdims=True)

    x = np.load(args.samples)

    # For a distance featurization, ask first whether the generated vectors
    # describe any 3-D arrangement at all, before asking whether it is a good
    # one. Nothing constrains the model to produce a valid distance matrix.
    if isinstance(feat, PairwiseDistance):
        eq = embedding_quality(feat, x)
        print("Euclidean validity of the generated distance vectors")
        print("-" * 52)
        print(f"{'variance in top 3 eigenvalues':<34s} {eq['frac_variance_in_top3']:>10.4f}")
        print(f"{'variance discarded by MDS':<34s} {eq['frac_variance_discarded']:>10.4f}")
        print(f"{'negative eigenvalue mass':<34s} {eq['neg_eig_mass']:>10.4f}")
        print(f"{'samples with negative eigenvalues':<34s} {eq['frac_with_neg_eig']:>10.4f}")
        print(f"{'worst negative eigenvalue':<34s} {eq['worst_neg_eig']:>10.4f}")
        print("\nAnything not in the top 3 eigenvalues is geometric information")
        print("describing a structure that does not fit in three dimensions. MDS")
        print("throws it away silently, so the conformers below look better than")
        print("the model's actual output.\n")

    gen = feat.inverse(x)
    print(f"{len(gen)} generated conformers, {gen.shape[1]} atoms\n")

    rep = geometry_report(gen, ref, z, bonds)
    print(f"{'metric':<24s} {'value':>10s}   {'good':>10s}")
    print("-" * 50)
    targets = {
        "frac_bonds_intact": ">0.95", "frac_no_clash": ">0.98", "frac_valid": ">0.95",
        "bond_mean_abs_err": "<0.02", "bond_std_ratio": "~1.0",
    }
    for k, v in rep.items():
        tgt = targets.get(k, "")
        print(f"{k:<24s} {v:>10.4f}   {tgt:>10s}")

    print(f"\n{'bond':<12s} {'ref mean':>10s} {'gen mean':>10s} {'ref std':>9s} {'gen std':>9s}")
    print("-" * 54)
    gb, rb = bond_lengths(gen, bonds), bond_lengths(ref, bonds)
    sym = {1: "H", 6: "C", 7: "N", 8: "O"}
    for k, (i, j) in enumerate(bonds):
        name = f"{sym.get(int(z[i]),'?')}{i}-{sym.get(int(z[j]),'?')}{j}"
        print(f"{name:<12s} {rb[:,k].mean():>10.3f} {gb[:,k].mean():>10.3f} "
              f"{rb[:,k].std():>9.3f} {gb[:,k].std():>9.3f}")

    print("\nHow to read this: matching the bond *means* is easy and says little. "
          "Matching the bond *standard deviations* is the real test -- it means the "
          "model captured the width of the thermal ensemble rather than collapsing "
          "onto the average structure.")


if __name__ == "__main__":
    main()
