"""Resolve the unresolved ILThermo compound names to structures via the NCI resolver.

§7 recorded "ILThermo + PubChem tops out near 350 pairs" and treated that as a
hard ceiling. It was a ceiling on *database lookup*: PubChem only returns
compounds someone deposited. ILThermo names are systematic
("1-butyl-3,5-dimethylpyridinium bromide"), so a name PARSER constructs them
directly from the morphology instead of looking them up.

NCI's Chemical Identifier Resolver does exactly that, needs no Java (unlike
OPSIN, which this box cannot install), and on a 20-name trial returned a SMILES
for 20/20 and a valid charge-balanced ion pair for 17/20.

Results are cached to disk so a re-run costs nothing, and requests are spaced to
be polite to a free public service.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
DATA = Path("data")
CACHE = DATA / "il_cir_cache.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--delay", type=float, default=0.75)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    import cirpy

    names = sorted(set(json.loads((DATA / "ilt_compounds.json").read_text()).values()))
    pub = json.loads((DATA / "il_pubchem_cache.json").read_text())
    todo = [n for n in names if not pub.get(n)]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [n for n in todo if n not in cache]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(names)} ILThermo names; {len(cache)} cached; {len(todo)} to resolve", flush=True)

    # A transient URLError must NOT be cached as a miss, or a re-run skips that
    # name forever. cirpy returns None for a genuine miss and raises for network
    # trouble, so only the former is recorded; the latter is retried with backoff
    # and then left out of the cache so the next run picks it up.
    errors = 0
    for i, n in enumerate(todo):
        got, err = None, None
        for attempt in range(3):
            try:
                got, err = cirpy.resolve(n, "smiles"), None
                break
            except Exception as exc:
                err = exc
                time.sleep(a.delay * (2 ** attempt))
        if err is None:
            cache[n] = got
        else:
            errors += 1
            if errors <= 5 or errors % 50 == 0:
                print(f"  ! {n[:44]}: {type(err).__name__} (uncached, will retry)", flush=True)
        time.sleep(a.delay)
        if (i + 1) % 100 == 0:
            CACHE.write_text(json.dumps(cache))
            hit = sum(1 for v in cache.values() if v)
            print(f"  {i+1}/{len(todo)}  resolved {hit}/{len(cache)} "
                  f"({hit/max(len(cache),1):.0%})", flush=True)
    CACHE.write_text(json.dumps(cache))
    if errors:
        print(f"\n{errors} names failed on network errors and were left uncached; "
              f"re-run to retry only those.", flush=True)

    # score what came back
    good = []
    for n, smi in cache.items():
        if not smi:
            continue
        fr = smi.split(".")
        if len(fr) != 2:
            continue
        ms = [Chem.MolFromSmiles(f) for f in fr]
        if any(m is None for m in ms):
            continue
        q = [Chem.GetFormalCharge(m) for m in ms]
        if sum(q) != 0 or not (max(q) > 0 and min(q) < 0):
            continue
        cat, an = (ms[0], ms[1]) if q[0] > 0 else (ms[1], ms[0])
        good.append({"name": n, "cation": Chem.MolToSmiles(cat), "anion": Chem.MolToSmiles(an),
                     "pair": f"{Chem.MolToSmiles(cat)}.{Chem.MolToSmiles(an)}",
                     "charge_cation": max(q), "charge_anion": min(q), "source": "ilthermo_cir"})
    (DATA / "il_cir_pairs.json").write_text(json.dumps(good, indent=1))
    print(f"\nresolved to valid ion pairs: {len(good):,} / {len(cache):,} attempted")
    print(f"-> {DATA/'il_cir_pairs.json'}")


if __name__ == "__main__":
    main()
