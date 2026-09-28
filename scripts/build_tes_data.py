"""Ion-level thermal-energy-storage evidence from MEASURED property data.

build_il_dataset_v2.py kept only melting points, but the Zenodo property files carry
values for decomposition temperature, heat capacity and thermal conductivity too --
exactly the quantities thermal energy storage is judged on.

Screening generated molecules with a trained property predictor would repeat the
mistake of 6.14: every such oracle here failed external validation (40.8 C MAE
against a 38.5 C constant baseline). So this builds a LOOKUP instead. For each ion,
aggregate the measured properties of every real IL containing it. A generated IL is
then scored by the measured behaviour of ILs that share its ions -- evidence from
data, not a model's opinion.

  liquid range = Tdec - MP, the span over which the salt is usable as a liquid heat
  store. The single most important TES figure of merit, and the one with the widest
  measured coverage (510 pairs).
"""
from __future__ import annotations
import collections, json
from pathlib import Path
import numpy as np

DATA = Path("data")


def _parse(fn, vcol, tcol=None, tlo=290.0, thi=320.0):
    out = collections.defaultdict(list)
    with open(DATA / f"{fn}.txt") as f:
        next(f)
        for ln in f:
            p = ln.split()
            if len(p) <= vcol:
                continue
            try:
                v = float(p[vcol])
                if tcol is not None and not (tlo <= float(p[tcol]) <= thi):
                    continue
            except ValueError:
                continue
            out[p[0]].append(v)
    return {k: float(np.median(v)) for k, v in out.items()}


def main() -> None:
    ca = {}
    for ln in open(DATA / "il_CA.smi"):
        p = ln.split()
        if len(p) >= 2:
            ca[p[1]] = p[0]

    MP = _parse("il_MP", 1)
    TD = _parse("il_TDECOMP", 1)
    HC = _parse("il_HEATCAPACITY", 3, 1)
    TC = _parse("il_THCOND", 2, 1)

    pairs = {}
    for k in set(MP) | set(TD) | set(HC) | set(TC):
        c, a = k.split("_")
        if c not in ca or a not in ca:
            continue
        rec = {"cation_smi": ca[c], "anion_smi": ca[a]}
        if k in MP: rec["mp_C"] = MP[k]
        if k in TD: rec["tdec_C"] = TD[k]
        if k in HC: rec["hcap_J_mol_K"] = HC[k]
        if k in TC: rec["thcond_W_m_K"] = TC[k]
        if "mp_C" in rec and "tdec_C" in rec:
            rec["liquid_range_C"] = rec["tdec_C"] - rec["mp_C"]
        pairs[k] = rec

    # ion-level aggregation: what do real ILs containing this ion actually do?
    ion = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, r in pairs.items():
        c, a = k.split("_")
        for side, i in (("cation", c), ("anion", a)):
            for f in ("mp_C", "tdec_C", "hcap_J_mol_K", "thcond_W_m_K", "liquid_range_C"):
                if f in r:
                    ion[i][f].append(r[f])
            ion[i]["_side"] = side
    ions = {}
    for i, d in ion.items():
        rec = {"smiles": ca[i], "side": d["_side"]}
        for f, v in d.items():
            if f.startswith("_"):
                continue
            rec[f] = float(np.median(v)); rec[f + "_n"] = len(v)
        ions[i] = rec

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    by_smi = {}
    for i, r in ions.items():
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None:
            continue
        by_smi[Chem.MolToSmiles(m)] = r | {"id": i}

    (DATA / "tes_pairs.json").write_text(json.dumps(pairs, indent=1))
    (DATA / "tes_ions.json").write_text(json.dumps(by_smi, indent=1))
    lr = np.array([r["liquid_range_C"] for r in pairs.values() if "liquid_range_C" in r])
    print(f"measured pairs {len(pairs):,}   ions {len(by_smi):,} (canonical SMILES keyed)")
    print(f"liquid range on {len(lr)} pairs: median {np.median(lr):.0f} C  "
          f"p75 {np.percentile(lr,75):.0f}  p90 {np.percentile(lr,90):.0f}  max {lr.max():.0f}")
    nlr = sum(1 for r in by_smi.values() if "liquid_range_C" in r)
    print(f"ions with a liquid-range record: {nlr}")
    print(f"-> {DATA/'tes_pairs.json'}, {DATA/'tes_ions.json'}")


if __name__ == "__main__":
    main()
