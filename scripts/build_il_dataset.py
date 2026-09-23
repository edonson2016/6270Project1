"""Build an ionic-liquid pair dataset: ILThermo names -> PubChem SMILES.

ILThermo lists ~2,000 pure ionic liquids by systematic name but publishes no
structures. PubChem resolves those names, and conveniently returns exactly the
dot-separated 'cation.anion' form, which is the representation we want.

Caches every lookup so the run is resumable; re-running costs nothing for names
already fetched.
"""
from __future__ import annotations

import json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

CATION_FAMILIES = ("imidazolium", "pyrrolidinium", "ammonium", "phosphonium",
                   "pyridinium", "sulfonium", "piperidinium", "morpholinium",
                   "guanidinium", "cholinium", "quinolinium", "thiazolium")
DATA = Path("data")
CACHE = DATA / "il_pubchem_cache.json"


def fetch_ilthermo() -> dict:
    f = DATA / "ilt1.json"
    if not f.exists():
        print("fetching ILThermo 1-component index ...")
        urllib.request.urlretrieve(
            "https://ilthermo.boulder.nist.gov/ILT2/ilsearch?cmp=&ncmp=1&year=&auth=&keyw=&prp=", f)
    rows = json.loads(f.read_text())["res"]
    names = {r[4]: r[8] for r in rows if r[4] and r[8]}
    return {k: v for k, v in names.items() if any(t in v.lower() for t in CATION_FAMILIES)}


def pubchem_smiles(name: str) -> str | None:
    """One request per call, with backoff on throttling.

    PubChem allows ~5 requests/second. The caller is responsible for pacing;
    this function makes exactly one request per attempt so the budget is
    countable, and backs off when PubChem says we are going too fast.
    """
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           + urllib.parse.quote(name) + "/property/CanonicalSMILES/TXT")
    for attempt in range(4):
        try:
            return urllib.request.urlopen(url, timeout=25).read().decode().strip().split("\n")[0]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None                      # genuinely unknown name
            time.sleep(1.5 * (2 ** attempt))     # 503/429: throttled, back off
        except Exception:
            time.sleep(1.0)
    return None


def normalize(name: str) -> list[str]:
    """IL names vary in bracket style between sources; try a few spellings."""
    variants = [name]
    v = name.replace("bis[(trifluoromethyl)sulfonyl]imide", "bis(trifluoromethylsulfonyl)imide")
    variants.append(v)
    variants.append(name.replace("[", "(").replace("]", ")"))
    variants.append(re.sub(r"\s+", " ", name).strip())
    seen, out = set(), []
    for x in variants:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def main() -> None:
    DATA.mkdir(exist_ok=True)
    names = fetch_ilthermo()
    print(f"{len(names)} ionic liquids named in ILThermo")
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    retry_failed = "--retry-failed" in sys.argv
    todo = [n for n in names.values()
            if n not in cache or (retry_failed and not cache[n])]
    print(f"{len(cache)} cached, {len(todo)} to fetch")

    for i, n in enumerate(todo):
        smi = None
        for v in normalize(n):
            smi = pubchem_smiles(v)
            time.sleep(0.25)          # pace EVERY request, not every name
            if smi:
                break
        cache[n] = smi
        if (i + 1) % 25 == 0:
            CACHE.write_text(json.dumps(cache))
            ok = sum(1 for v in cache.values() if v)
            print(f"  {i+1}/{len(todo)}  resolved {ok}/{len(cache)} "
                  f"({ok/max(len(cache),1):.0%})", flush=True)
        time.sleep(0.22)   # PubChem asks for <=5 requests/second
    CACHE.write_text(json.dumps(cache))

    # Keep only genuine two-fragment salts with opposite formal charges.
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    pairs, cats, ans = [], set(), set()
    for name, smi in cache.items():
        if not smi or "." not in smi:
            continue
        frags = smi.split(".")
        if len(frags) != 2:
            continue
        mols = [Chem.MolFromSmiles(f) for f in frags]
        if any(m is None for m in mols):
            continue
        q = [Chem.GetFormalCharge(m) for m in mols]
        if not (max(q) > 0 and min(q) < 0):
            continue
        cat = Chem.MolToSmiles(mols[0] if q[0] > 0 else mols[1])
        an = Chem.MolToSmiles(mols[1] if q[0] > 0 else mols[0])
        pairs.append({"name": name, "cation": cat, "anion": an,
                      "pair": f"{cat}.{an}", "charge_cation": max(q), "charge_anion": min(q)})
        cats.add(cat); ans.add(an)

    seen, uniq = set(), []
    for p in pairs:
        if p["pair"] not in seen:
            seen.add(p["pair"]); uniq.append(p)

    (DATA / "il_pairs.json").write_text(json.dumps(uniq, indent=2))
    (DATA / "il_pairs.txt").write_text("\n".join(p["pair"] for p in uniq))
    print(f"\n{len(uniq)} unique IL pairs  |  {len(cats)} unique cations  |  {len(ans)} unique anions")
    print(f"observed pairs are {len(uniq)/max(len(cats)*len(ans),1):.2%} of the "
          f"{len(cats)*len(ans):,} possible combinations")
    print(f"wrote {DATA/'il_pairs.json'}")


if __name__ == "__main__":
    main()
