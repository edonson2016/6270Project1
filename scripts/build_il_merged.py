"""Merge the CIR-resolved ILThermo pairs into the observed corpus, TRAIN ONLY.

The point of this corpus is to test whether more observed ILs reduce the flow's
overfitting (§6.5: the flow sits 2.3-3.1x closer to train latents than held-out
ones). That is only measurable if the evaluation sets do not move, so:

  * the first 4,790 entries are the Zenodo corpus in its ORIGINAL ORDER, so
    make_splits(seed=0) reproduces valid / test / test_ion exactly;
  * every new pair is appended after them and assigned to TRAIN;
  * any new pair whose cation belongs to test_ion is DROPPED, because adding it
    would destroy the cation-disjointness that makes test_ion a generalization
    test at all;
  * any new pair that duplicates a held-out pair is likewise dropped.

So the only thing that changes between this run and §6.9 is the size of the
flow's training set. Everything else -- decoder curriculum, valid, test,
test_ion -- is held fixed.
"""
from __future__ import annotations
import json
from pathlib import Path

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm.il_eval import make_splits, ion_sets

DATA = Path("data")


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None


def main() -> None:
    zen = json.loads((DATA / "il_pairs_v2.json").read_text())
    sp = make_splits(zen, seed=0)
    heldP = set()
    for k in ("valid", "test", "test_ion"):
        heldP |= ion_sets(zen, sp[k])[0]
    heldC = ion_sets(zen, sp["test_ion"])[1]
    trainP = ion_sets(zen, sp["train"])[0]
    zenP = {f"{canon(p['cation'])}.{canon(p['anion'])}" for p in zen}

    cache = json.loads((DATA / "il_cir_cache.json").read_text())
    kept, drop_dup, drop_held, drop_cat, bad = [], 0, 0, 0, 0
    seen = set()
    for name, smi in cache.items():
        if not smi:
            continue
        fr = smi.split(".")
        if len(fr) != 2:
            bad += 1; continue
        ms = [Chem.MolFromSmiles(f) for f in fr]
        if any(m is None for m in ms):
            bad += 1; continue
        q = [Chem.GetFormalCharge(m) for m in ms]
        if sum(q) != 0 or not (max(q) > 0 and min(q) < 0):
            bad += 1; continue
        cat, an = (ms[0], ms[1]) if q[0] > 0 else (ms[1], ms[0])
        c, a = Chem.MolToSmiles(cat), Chem.MolToSmiles(an)
        pair = f"{c}.{a}"
        if pair in zenP or pair in seen:
            drop_dup += 1; continue
        if pair in heldP:
            drop_held += 1; continue
        if c in heldC:                      # would break test_ion's cation-disjointness
            drop_cat += 1; continue
        seen.add(pair)
        kept.append({"name": name, "cation": c, "anion": a, "pair": pair,
                     "charge_cation": max(q), "charge_anion": min(q),
                     "properties": [], "melting_point_C": None, "source": "ilthermo_cir"})

    merged = list(zen) + kept
    (DATA / "il_pairs_v3.json").write_text(json.dumps(merged))
    (DATA / "il_smi_ils_new.txt").write_text("\n".join(p["pair"] for p in kept))
    print(f"cache entries              : {len(cache):,}")
    print(f"  unusable / not an IL     : {bad}")
    print(f"  duplicate of corpus      : {drop_dup}")
    print(f"  duplicate of a held-out  : {drop_held}")
    print(f"  test_ion cation (dropped): {drop_cat}")
    print(f"  KEPT, appended to train  : {len(kept)}")
    print(f"\ncorpus {len(zen):,} -> {len(merged):,}  (train {len(trainP):,} -> "
          f"{len(trainP)+len(kept):,}, +{len(kept)/len(trainP):.1%})")
    print(f"-> {DATA/'il_pairs_v3.json'}  and  {DATA/'il_smi_ils_new.txt'}")


if __name__ == "__main__":
    main()
