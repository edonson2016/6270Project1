"""Screen generated ionic liquids for thermal-energy-storage potential.

Scored by LOOKUP against measured data, not by a trained predictor: every property
oracle built for this project failed external validation (6.14), so a generated IL is
judged by what real ILs sharing its ions actually do.

  liquid_range = Tdec - MP, the usable span of a liquid heat store, and the TES
  figure of merit with the widest measured coverage (510 pairs, 329 ions).

A candidate needs BOTH ions to carry a measured record. That is a hard evidence
requirement and it rejects most novel chemistry -- deliberately, because an
unmeasured ion has no thermal evidence at all, and pretending otherwise is what the
retired oracle did.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from fm.il_eval import (make_splits, ion_sets, plausibility, element_set, heavy_and_mw,
                        repair_radicals, novelty, _family, _CAT_PAT, _AN_PAT)

DATA = Path("data")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("arms", nargs="+", help="label=samplefile")
    p.add_argument("--out", default="runs/il/results/tes_screen.json")
    a = p.parse_args()

    ions = json.loads((DATA / "tes_ions.json").read_text())
    ils = json.loads((DATA / "il_pairs_v2.json").read_text())
    sp = make_splits(ils, seed=0)
    P, C, A = ion_sets(ils, sp["train"])
    corpusP = {q["pair"] for q in ils}
    allC = {x for k in sp for x in ion_sets(ils, sp[k])[1]}
    allA = {x for k in sp for x in ion_sets(ils, sp[k])[2]}
    ts = [ils[i]["pair"] for i in sp["train"]]
    ELEM = element_set(ts); RH, RM = heavy_and_mw(ts)

    out = {}
    for spec in a.arms:
        label, path = spec.split("=", 1)
        smis = [repair_radicals(x) for x in open(path).read().split("\n") if x]
        n = len(smis)
        m, pairs = plausibility(smis, n, ELEM, RH, RM)
        m |= novelty(pairs, P, C, A)
        seen, cands = set(), []
        for pr in pairs:
            if pr in seen:
                continue
            seen.add(pr)
            c, an = pr.split(".")
            rc, ra = ions.get(c), ions.get(an)
            rec = {"pair": pr, "cation": c, "anion": an,
                   "novel_pair": pr not in corpusP,
                   "novel_cation": c not in allC, "novel_anion": an not in allA,
                   "cation_measured": rc is not None, "anion_measured": ra is not None}
            lrs = [r["liquid_range_C"] for r in (rc, ra) if r and "liquid_range_C" in r]
            tds = [r["tdec_C"] for r in (rc, ra) if r and "tdec_C" in r]
            mps = [r["mp_C"] for r in (rc, ra) if r and "mp_C" in r]
            hcs = [r["hcap_J_mol_K"] for r in (rc, ra) if r and "hcap_J_mol_K" in r]
            if lrs: rec["liquid_range_C"] = float(np.mean(lrs)); rec["lr_n_ions"] = len(lrs)
            if tds: rec["tdec_C"] = float(np.mean(tds))
            if mps: rec["mp_C"] = float(np.mean(mps))
            if hcs: rec["hcap_J_mol_K"] = float(np.mean(hcs))
            cm = Chem.MolFromSmiles(c)
            rec["cation_family"] = (_family(cm, _CAT_PAT) or "none") if cm else "none"
            am = Chem.MolFromSmiles(an)
            rec["anion_family"] = (_family(am, _AN_PAT) or "none") if am else "none"
            mol = Chem.MolFromSmiles(pr)
            rec["heavy"] = mol.GetNumHeavyAtoms() if mol else None
            cands.append(rec)

        ev = [c for c in cands if c.get("lr_n_ions") == 2]
        ev.sort(key=lambda r: -r["liquid_range_C"])
        funnel = {
            "sampled": n,
            "plausible": int(round(m["plausible"] * n)),
            "distinct": len(cands),
            "both_ions_measured": sum(1 for c in cands if c["cation_measured"] and c["anion_measured"]),
            "liquid_range_both_ions": len(ev),
            "lr_gt_250": sum(1 for c in ev if c["liquid_range_C"] > 250),
            "lr_gt_300": sum(1 for c in ev if c["liquid_range_C"] > 300),
            "novel_pair_and_lr_gt_250": sum(1 for c in ev
                                            if c["novel_pair"] and c["liquid_range_C"] > 250),
        }
        out[label] = {"funnel": funnel, "candidates": cands[:4000], "top": ev[:25],
                      "plausible_rate": m["plausible"], "uniqueness": m["uniqueness"],
                      "novel_cation_rate": m["novel_cation_rate"]}
        print(f"\n=== {label} ===")
        for k, v in funnel.items():
            print(f"  {k:26} {v:6d}" + (f"  ({v/n:5.1%})" if isinstance(v, int) else ""))
        if ev:
            print(f"  best liquid range: {ev[0]['liquid_range_C']:.0f} C  "
                  f"{ev[0]['cation_family']}/{ev[0]['anion_family']}  "
                  f"novel_pair={ev[0]['novel_pair']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
