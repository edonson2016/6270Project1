"""Does the latent preserve a symmetric double well?

Malonaldehyde's aldehyde torsion is a symmetric double well, so the correct
population split is exactly 50/50 BY SYMMETRY. That makes it a sharper
mode-coverage test than any asymmetric system: there is no ambiguity about the
right answer, and a model that collapses toward one well shows up immediately
as a broken symmetry.

(Note: rMD17 malonaldehyde is the keto tautomer and shows NO proton transfer --
the minimum O-H distance over 100k frames is 1.74 A. The torsions are the
bimodal coordinate here, not a transferring proton.)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fm import Standardizer, sample
from fm.featurize import AlignedCartesian
from scripts.exp2_conformer_pareto import train_ae, run_fm


def dihedral(c, i, j, k, l):
    b0 = c[:, i]-c[:, j]; b1 = c[:, k]-c[:, j]; b2 = c[:, l]-c[:, k]
    b1 = b1/np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0-(b0*b1).sum(1, keepdims=True)*b1
    w = b2-(b2*b1).sum(1, keepdims=True)*b1
    return np.degrees(np.arctan2((np.cross(b1, v)*w).sum(1), (v*w).sum(1)))


def modes(c):
    """Population split of the symmetric torsional double well."""
    t = dihedral(c, 4, 0, 1, 2)
    return float((t > 0).mean()), float((t <= 0).mean())


def main():
    dev = torch.device("cpu"); torch.manual_seed(0)
    raw = np.load("data/rmd17_malonaldehyde.npz")
    coords = raw["coords"][:90000]
    F = AlignedCartesian().fit(coords)
    x = torch.as_tensor(F.transform(coords)); sc = Standardizer().fit(x); xs = sc.transform(x)

    p, m = modes(coords)
    print(f"\nREFERENCE rMD17     well+ {p:.4f}  well- {m:.4f}   "
          f"(symmetry says 0.5000/0.5000, |skew| {abs(p-0.5):.4f})\n")
    rows = [("reference", None, p, m)]

    print("handcrafted (no autoencoder)")
    ema, _, _ = run_fm(xs, 4000, 256, 4, dev)
    g = sample(ema, n=8000, shape=(xs.shape[1],), n_steps=50, method="heun", device=dev)
    gc = F.inverse(sc.inverse(g.cpu()).numpy())
    p, m = modes(gc)
    print(f"  aligned Cartesian   well+ {p:.4f}  well- {m:.4f}   |skew| {abs(p-0.5):.4f}")
    rows.append(("handcrafted:aligned", None, p, m))

    for d in (5, 10, 16, 21, 26):
        ae, recon, _, _ = train_ae(xs, d, 15, 256, dev)
        with torch.no_grad():
            lat = torch.cat([ae.encode(xs[i:i+4096].to(dev))[0].cpu()
                             for i in range(0, len(xs), 4096)])
        lsc = Standardizer().fit(lat)
        ema, _, _ = run_fm(lsc.transform(lat), 4000, 256, 4, dev)
        gz = sample(ema, n=8000, shape=(d,), n_steps=50, method="heun", device=dev)
        with torch.no_grad():
            xh = ae.decode(lsc.inverse(gz.cpu()).to(dev)).cpu()
        gc = F.inverse(sc.inverse(xh).numpy())
        p, m = modes(gc)
        print(f"  latent d={d:<3d}         well+ {p:.4f}  well- {m:.4f}   "
              f"|skew| {abs(p-0.5):.4f}   (AE recon {recon:.5f})", flush=True)
        rows.append((f"latent d={d}", recon, p, m))

    np.save("runs/overnight/malon_modes.npy", np.array([(r[2], r[3]) for r in rows]))
    print("\n|skew| is the number that matters: 0 means both wells equally populated,")
    print("0.5 means total collapse onto one well.")


if __name__ == "__main__":
    main()
