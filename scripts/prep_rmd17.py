"""Download an rMD17 molecule and featurize it into (N, D) arrays for flow matching.

rMD17 is 100k DFT-computed conformations per molecule from an MD trajectory --
a thermalized conformational ensemble of a fixed-composition system. That is
structurally the same problem as an ion-pair ensemble, but it is a 65 MB
download with energies and forces attached, so you can actually check your work.

    .venv/bin/python scripts/prep_rmd17.py --molecule ethanol
    .venv/bin/python scripts/prep_rmd17.py --molecule malonaldehyde --featurizer distance
"""

from __future__ import annotations

import argparse
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fm.featurize import AlignedCartesian, PairwiseDistance, infer_bonds  # noqa: E402

# figshare file ids for the per-molecule .npz files of doi 10.6084/m9.figshare.12672038
FILES = {
    "ethanol": "62265733",
    "malonaldehyde": "62265736",
    "benzene": "62265739",
    "toluene": "62265742",
    "naphthalene": "62265751",
    "azobenzene": "62265754",
    "aspirin": "62265757",
    "paracetamol": "62265760",
}
URL = "https://ndownloader.figshare.com/files/{}"


def download(molecule: str, data_dir: Path) -> Path:
    path = data_dir / f"rmd17_{molecule}.npz"
    if path.exists():
        print(f"using cached {path}")
        return path
    if molecule not in FILES:
        raise SystemExit(f"unknown molecule {molecule!r}; known: {sorted(FILES)}")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {molecule} ...")
    urllib.request.urlretrieve(URL.format(FILES[molecule]), path)
    print(f"  -> {path} ({path.stat().st_size/1e6:.0f} MB)")
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--molecule", default="ethanol", choices=sorted(FILES))
    p.add_argument("--featurizer", default="aligned", choices=["aligned", "distance"])
    p.add_argument("--n", type=int, default=100_000, help="how many conformers to use")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    raw = np.load(download(args.molecule, data_dir))
    coords = raw["coords"][: args.n]
    z = raw["nuclear_charges"]
    energies = raw["energies"][: args.n]

    print(f"\n{args.molecule}: {len(coords)} conformers, {len(z)} atoms, Z={list(z)}")
    print(f"energy spread: {energies.std():.2f} kcal/mol")

    feat = AlignedCartesian() if args.featurizer == "aligned" else PairwiseDistance()
    feat.fit(coords)
    x = feat.transform(coords)
    print(f"featurizer={args.featurizer}  D={x.shape[1]}")

    # Round-trip error tells you how much the featurization itself costs you,
    # before the model has made a single mistake. Check this first, always.
    recon = feat.inverse(x[:200])
    ref = coords[:200] - coords[:200].mean(axis=1, keepdims=True)
    if args.featurizer == "aligned":
        err = np.abs(recon - feat.transform(coords[:200]).reshape(-1, len(z), 3)).max()
    else:
        from fm.featurize import kabsch_rotation

        def align_err(r, c):
            r = r - r.mean(0)
            return np.abs(r @ kabsch_rotation(r, c).T - c).max()

        # MDS recovers a structure only up to reflection, so the round-trip has
        # to be scored against both chiralities. The fraction needing a mirror
        # is not noise -- it is the featurization telling you it cannot
        # represent chirality at all.
        errs, n_mirror = [], 0
        for r, c in zip(recon, ref):
            e_direct = align_err(r, c)
            mirrored = r.copy()
            mirrored[:, 0] *= -1
            e_mirror = align_err(mirrored, c)
            errs.append(min(e_direct, e_mirror))
            n_mirror += int(e_mirror < e_direct)
        err = float(np.max(errs))
        print(f"round-trip needed mirroring for {n_mirror}/{len(recon)} structures "
              f"-- distances cannot encode chirality")
    print(f"round-trip max atom error: {err:.4f} A")

    stem = data_dir / f"{args.molecule}_{args.featurizer}"
    np.save(f"{stem}_x.npy", x)
    np.save(f"{stem}_energies.npy", energies.astype(np.float32))
    bonds = infer_bonds(coords, z)
    with open(f"{stem}_meta.pkl", "wb") as f:
        pickle.dump({"featurizer": feat, "z": z, "bonds": bonds,
                     "coords_sample": coords[:5000]}, f)
    print(f"\nwrote {stem}_x.npy  shape={x.shape}")
    print(f"      {stem}_meta.pkl  ({len(bonds)} bonds inferred)")
    print(f"\nnext:\n  .venv/bin/python scripts/train_vectors.py --x1 {stem}_x.npy --out runs/{args.molecule}")


if __name__ == "__main__":
    main()
