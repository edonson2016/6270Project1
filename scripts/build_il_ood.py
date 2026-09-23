"""Build the out-of-distribution IL test set from the ILThermo/PubChem branch.

`data/il_pairs.json` is the leftover of the dead end documented in the docs:
ILThermo names ~2,072 pure ILs, PubChem resolves ~17% of them, and the result
tops out near 350 pairs -- too few to train on, which is why the Zenodo record
became the training corpus instead.

Too few to train on is not too few to *test* on. It is a genuinely different
curation pipeline (different structure source, different canonical SMILES), so
whatever survives deduplication against Zenodo is a real held-out set that no
part of the training pipeline has seen.

Two tiers come out of this, and they must be read separately:

  all_ood      every ILThermo pair absent from Zenodo. Mostly ordinary
               imidazolium/NTf2-family ILs -- a different SOURCE, but largely
               the same DISTRIBUTION. Tests curation/representation robustness.
  novel_ion    the subset containing a cation or anion never seen in Zenodo.
               This is the only tier that is out of distribution chemically.
"""
from __future__ import annotations
import json
from pathlib import Path
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
DATA = Path("data")


def canon(s: str) -> str | None:
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None


def main() -> None:
    zen = json.loads((DATA / "il_pairs_v2.json").read_text())
    ilt = json.loads((DATA / "il_pairs.json").read_text())

    zc = {canon(p["cation"]) for p in zen} - {None}
    za = {canon(p["anion"]) for p in zen} - {None}
    zp = {f"{canon(p['cation'])}.{canon(p['anion'])}" for p in zen}

    out, seen = [], set()
    for p in ilt:
        c, a = canon(p["cation"]), canon(p["anion"])
        if c is None or a is None:
            continue
        pair = f"{c}.{a}"
        if pair in zp or pair in seen:      # drop Zenodo overlap and internal dupes
            continue
        seen.add(pair)
        out.append({"name": p.get("name", ""), "cation": c, "anion": a, "pair": pair,
                    "charge_cation": p.get("charge_cation"),
                    "charge_anion": p.get("charge_anion"),
                    "novel_cation": c not in zc, "novel_anion": a not in za})

    novel = [p for p in out if p["novel_cation"] or p["novel_anion"]]
    (DATA / "il_ood_pairs.json").write_text(json.dumps(out, indent=1))
    print(f"ILThermo pairs read        : {len(ilt)}")
    print(f"OOD pairs after dedup      : {len(out)}")
    print(f"  with an unseen cation    : {sum(p['novel_cation'] for p in out)}")
    print(f"  with an unseen anion     : {sum(p['novel_anion'] for p in out)}")
    print(f"  novel_ion tier (either)  : {len(novel)}")
    print(f"-> {DATA/'il_ood_pairs.json'}")


if __name__ == "__main__":
    main()
