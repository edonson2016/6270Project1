"""Move part of the ILThermo OOD set into training, hold the rest out.

Every run so far has shown `sw2_ood` flat at 0.46-0.55 across every model,
sampler and latent dimension: the flow puts essentially no mass where the
ILThermo latents live. §6.7 established that those latents are NOT off-manifold
-- same subspace, same radius -- but shifted 0.30-0.58 SD within it.

What no run has answered is whether the flow COULD cover that region if shown
part of it. That separates a coverage failure from a capacity one, so:

  train    the first `--n-train` OOD pairs are appended to the training corpus
  heldout  the remainder becomes the new OOD evaluation set

Two readings come out of it. `sw2` on the held-out remainder says whether
partial exposure generalizes across the shift. `test` / `test_ion` say what the
injection COSTS the main distribution -- a shifted sub-population could dilute it.

Base corpus is selectable so the same split can be re-applied on top of the
CIR-extended corpus later.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm.il_eval import make_splits, make_splits_extended, ion_sets

DATA = Path("data")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="il_pairs_v2")
    p.add_argument("--base-emb", default="il_emb_ils")
    p.add_argument("--base-smi", default="il_smi_ils")
    p.add_argument("--n-original", type=int, default=0, help="if the base is itself extended")
    p.add_argument("--n-train", type=int, default=150)
    p.add_argument("--out", default="v4")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    base = json.loads((DATA / f"{a.base}.json").read_text())
    n_orig = a.n_original or len(base)
    sp = (make_splits_extended(base, a.n_original, seed=a.seed) if a.n_original
          else make_splits(base, seed=a.seed))
    heldC = ion_sets(base, sp["test_ion"])[1]
    basePairs = {q["pair"] for q in base}

    ood = json.loads((DATA / "il_ood_pairs.json").read_text())
    Eo = np.load(DATA / "il_emb_ood.npy")
    assert len(ood) == len(Eo)

    # a pair whose cation sits in test_ion cannot go to train without destroying
    # the cation-disjointness that makes test_ion a generalization test
    ok = [i for i, q in enumerate(ood)
          if q["cation"] not in heldC and q["pair"] not in basePairs]
    blocked = len(ood) - len(ok)
    rng = np.random.default_rng(a.seed); rng.shuffle(ok)
    tr_idx, ho_idx = ok[: a.n_train], ok[a.n_train:]
    print(f"OOD pairs {len(ood)}   eligible for train {len(ok)}   blocked {blocked}")
    print(f"  -> {len(tr_idx)} into TRAIN, {len(ho_idx)} held out as the OOD eval set")

    merged = list(base) + [dict(ood[i], properties=[], melting_point_C=None,
                                source="ilthermo_ood_train") for i in tr_idx]
    Eb = np.load(DATA / f"{a.base_emb}.npy")
    Sb = (DATA / f"{a.base_smi}.txt").read_text().split("\n")
    E = np.concatenate([Eb, Eo[tr_idx]])
    S = Sb + [ood[i]["pair"] for i in tr_idx]
    assert len(E) == len(S) == len(merged)

    (DATA / f"il_pairs_{a.out}.json").write_text(json.dumps(merged))
    np.save(DATA / f"il_emb_ils_{a.out}.npy", E)
    (DATA / f"il_smi_ils_{a.out}.txt").write_text("\n".join(S))
    (DATA / f"il_ood_pairs_{a.out}.json").write_text(json.dumps([ood[i] for i in ho_idx], indent=1))
    np.save(DATA / f"il_emb_ood_{a.out}.npy", Eo[ho_idx])

    # the appended pairs must land in train, and the splits must not move
    sp2 = make_splits_extended(merged, n_orig, seed=a.seed)
    for k in ("valid", "test", "test_ion"):
        assert np.array_equal(sp[k], sp2[k]), f"{k} split moved"
    print(f"corpus {len(base):,} -> {len(merged):,}   train {len(sp['train']):,} -> "
          f"{len(sp2['train']):,}  (+{len(tr_idx)/len(sp['train']):.1%})")
    print(f"  valid/test/test_ion unchanged: True")
    print(f"-> il_pairs_{a.out}.json, il_emb_ils_{a.out}.npy, il_ood_pairs_{a.out}.json "
          f"({len(ho_idx)} held out)")


if __name__ == "__main__":
    main()
