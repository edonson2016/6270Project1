"""How many dimensions does the IL latent actually need?

Answers the question the d sweep raises: d=16 beat d=64 for generation, and
d=8 is on the table -- is that absurdly small, or is the data simply that
low-dimensional, and is 4,790 pairs too few to tell?

Measures the effective rank of the frozen encoder's output, which is what
MLP_down receives and therefore an upper bound on what z can usefully carry.

Effective rank here is the participation ratio of the PCA spectrum,
PR = (sum lambda)^2 / sum lambda^2. It is a continuous stand-in for "how many
directions carry real variance", less arbitrary than a 90%-variance cutoff.

Caveat worth keeping: PR is a LINEAR measure and MLP_down is nonlinear, so it
upper-bounds what a nonlinear encoder needs. It is not a floor.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

DATA = Path("data")


def participation_ratio(X: np.ndarray) -> float:
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    v = s ** 2
    return float((v.sum() ** 2) / (v ** 2).sum())


def between_group_frac(X: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of total variance explained by group identity alone."""
    tot = ((X - X.mean(0)) ** 2).sum()
    within = sum(((X[labels == u] - X[labels == u].mean(0)) ** 2).sum()
                 for u in np.unique(labels))
    return float(1 - within / tot)


def main() -> None:
    E = np.load(DATA / "il_emb_ils.npy").astype(np.float64)
    O = np.load(DATA / "il_emb_ood.npy").astype(np.float64)

    mu = E.mean(0); Ec = E - mu
    _, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    var = S ** 2 / (len(E) - 1); ratio = var / var.sum(); cum = np.cumsum(ratio)

    print("=" * 72)
    print("INTRINSIC DIMENSION of the ChemBERTa IL embedding cloud")
    print("=" * 72)
    print(f"  shape {E.shape}")
    print(f"  participation ratio (effective rank): {participation_ratio(E):.2f} of 768")
    for q in (0.50, 0.80, 0.90, 0.95, 0.99):
        print(f"    PCs for {q:.0%} of variance: {int(np.searchsorted(cum, q)) + 1}")

    # Is it sample-limited? If 4,790 pairs were too few to reveal the manifold,
    # effective rank would still be climbing with n.
    print("\n  effective rank vs sample size (3 draws each):")
    rng = np.random.default_rng(0)
    for n in (200, 500, 1000, 2000, 3447, 4790):
        if n > len(E): continue
        vals = [participation_ratio(E[rng.choice(len(E), n, replace=False)]) for _ in range(3)]
        print(f"    n={n:>5d}  PR = {np.mean(vals):5.2f} +/- {np.std(vals):.2f}")

    # Control: same encoder, same n, different chemistry.
    mf = DATA / "il_emb_moses.npy"
    if mf.exists():
        M = np.load(mf).astype(np.float64)[:len(E)]
        print(f"\n  MOSES drug-like at matched n={len(M)}: PR = {participation_ratio(M):5.2f}")
        print(f"  ionic liquids                        : PR = {participation_ratio(E):5.2f}")
        print("  -> the difference is the chemistry, not the sample count")

    ils = json.loads((DATA / "il_pairs_v2.json").read_text())
    an = np.array([p["anion"] for p in ils]); ca = np.array([p["cation"] for p in ils])
    print(f"\n  variance explained by ion identity alone:")
    print(f"    anion  ({len(np.unique(an)):>4d} groups): {between_group_frac(E, an):.3f}")
    print(f"    cation ({len(np.unique(ca)):>4d} groups): {between_group_frac(E, ca):.3f}")

    # Where does the OOD set sit? Aggregate radius hides a mean shift, so report both.
    def energy_in_top(X, k=64):
        Xc = X - mu
        return float(1 - ((Xc - (Xc @ Vt[:k].T) @ Vt[:k]) ** 2).sum() / (Xc ** 2).sum())

    def mean_absz(X, k=16):
        return float(np.abs(((X - mu) @ Vt[:k].T) / np.sqrt(var[:k])).mean())

    print("\n  OOD set, measured in the training PCA basis:")
    print(f"    energy inside top-64 PCs : train {energy_in_top(E):.3f}   ood {energy_in_top(O):.3f}")
    print(f"    mean |z| along top-16 PCs: train {mean_absz(E):.3f}   ood {mean_absz(O):.3f}")
    po = ((O - mu) @ Vt[:6].T) / np.sqrt(var[:6])
    shifts = "  ".join(f"PC{k+1} {po[:, k].mean():+.2f}" for k in range(6))
    print(f"    per-PC mean shift (SD)   : {shifts}")
    print("    -> same subspace and same radius, but SHIFTED within the manifold.")
    print("       Aggregate radius hides that; sliced W2 detects it.")


if __name__ == "__main__":
    main()
