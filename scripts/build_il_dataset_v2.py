"""Build the ionic-liquid pair dataset from the Zenodo IL properties collection.

Supersedes the ILThermo+PubChem route, which topped out near 350 pairs because
most IL names in ILThermo's long tail are simply not in PubChem (523 of 785
cation names appear exactly once, so there is no popular-ion shortcut either).

This source ships what we actually need directly:
  CA.smi         ion id -> SMILES, for both cations (C####) and anions (A####)
  <PROPERTY>.txt observed CATION_ANION pairs, one file per measured property

Unioning the pair lists across all property files gives every experimentally
attested combination, and MP.txt additionally carries melting points, which
makes a room-temperature filter possible later without re-sourcing anything.

    .venv/bin/python scripts/build_il_dataset_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path("data")
PROPS = ["MP", "CO2CAPACITY", "CONDUCTIVITY", "CYTOTOXICITY", "DENSITY", "GTT",
         "HEATCAPACITY", "RI", "SURFACETENSION", "TDECOMP", "THCOND", "VISCOSITY"]


def main() -> None:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    # ---- ion inventory
    ions: dict[str, str] = {}
    for line in (DATA / "il_CA.smi").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            ions[parts[1].strip()] = parts[0].strip()
    n_cat = sum(1 for k in ions if k.startswith("C"))
    n_an = sum(1 for k in ions if k.startswith("A"))
    print(f"ion inventory: {len(ions)} ions  ({n_cat} cations, {n_an} anions)")

    # ---- observed pairs, unioned over every property file
    pair_props: dict[tuple[str, str], set[str]] = {}
    mp: dict[tuple[str, str], float] = {}
    for p in PROPS:
        f = DATA / f"il_{p}.txt"
        if not f.exists():
            continue
        lines = f.read_text().splitlines()[1:]
        n = 0
        for line in lines:
            tok = line.split()
            if not tok or "_" not in tok[0]:
                continue
            c, a = tok[0].split("_", 1)
            pair_props.setdefault((c, a), set()).add(p)
            n += 1
            if p == "MP" and len(tok) > 1:
                try:
                    mp[(c, a)] = float(tok[1])
                except ValueError:
                    pass
        print(f"  {p:<16s} {n:>6d} records")
    print(f"\nunique observed pairs: {len(pair_props)}")

    # ---- assemble, validating chemistry
    out, bad_id, bad_parse, bad_charge = [], 0, 0, 0
    cats, ans = set(), set()
    for (c, a), props in sorted(pair_props.items()):
        if c not in ions or a not in ions:
            bad_id += 1
            continue
        cs, as_ = ions[c], ions[a]
        mc, ma = Chem.MolFromSmiles(cs), Chem.MolFromSmiles(as_)
        if mc is None or ma is None:
            bad_parse += 1
            continue
        qc, qa = Chem.GetFormalCharge(mc), Chem.GetFormalCharge(ma)
        if not (qc > 0 and qa < 0):
            bad_charge += 1
            continue
        cc, ca = Chem.MolToSmiles(mc), Chem.MolToSmiles(ma)
        rec = {"cation_id": c, "anion_id": a, "cation": cc, "anion": ca,
               "pair": f"{cc}.{ca}", "charge_cation": qc, "charge_anion": qa,
               "n_heavy": mc.GetNumHeavyAtoms() + ma.GetNumHeavyAtoms(),
               "properties": sorted(props)}
        if (c, a) in mp:
            rec["melting_point_C"] = mp[(c, a)]
        out.append(rec)
        cats.add(cc); ans.add(ca)

    print(f"dropped: {bad_id} unknown ion id, {bad_parse} unparseable, {bad_charge} wrong charge")
    print(f"\nFINAL: {len(out)} IL pairs | {len(cats)} cations | {len(ans)} anions")
    grid = len(cats) * len(ans)
    print(f"  observed pairs are {len(out)/grid:.2%} of the {grid:,} possible combinations")
    with_mp = [r for r in out if "melting_point_C" in r]
    print(f"  with melting point: {len(with_mp)}")
    if with_mp:
        rt = [r for r in with_mp if r["melting_point_C"] < 25]
        print(f"  liquid at room temperature (mp < 25 C): {len(rt)} ({len(rt)/len(with_mp):.0%})")
    lens = sorted(len(r["pair"]) for r in out)
    print(f"  pair SMILES length: median {lens[len(lens)//2]}, "
          f"p95 {lens[int(len(lens)*0.95)]}, max {lens[-1]}")

    (DATA / "il_pairs_v2.json").write_text(json.dumps(out, indent=2))
    (DATA / "il_pairs_v2.txt").write_text("\n".join(r["pair"] for r in out))
    print(f"\nwrote {DATA/'il_pairs_v2.json'} and .txt")


if __name__ == "__main__":
    main()
