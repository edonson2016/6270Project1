"""Fetch measured normal melting temperatures from NIST ILThermo.

Everything validating melting point in this project so far has been a model
fitted to the SAME 1,588 Zenodo labels the flow was conditioned on -- the flow
learned p(z|MP) and the oracle learned MP=f(z) from one dataset, so a spurious
correlation in those labels would fool both at once. ILThermo is a separate
compilation by a separate group, which breaks that circularity.

Two distinct uses come out of it, and the second is the one nobody has done:

  external   compounds ILThermo has and the Zenodo corpus does not. The oracle
             never saw them and neither did the flow, so they test whether the
             oracle predicts chemistry or merely memorized corpus structure.

  overlap    compounds in BOTH. Two independent compilations of the same salt.
             Their disagreement is the irreducible label noise -- the floor
             below which no predictor can be judged, and which nothing in this
             project has measured.

Values are cached by setid so a re-run is free, requests are spaced to be polite
to a public service, and a plain User-Agent is set because NIST returns 403 to
urllib's default.
"""
from __future__ import annotations
import json, time, urllib.request, urllib.error
from pathlib import Path

DATA = Path("data"); CACHE = DATA / "ilt_mp_cache.json"
UA = {"User-Agent": "fm-chem research script (contact edonson2016@gmail.com)",
      "Referer": "https://ilthermo.boulder.nist.gov/"}
PROP = "Normal melting temperature"


def fetch_set(sid: str):
    url = f"https://ilthermo.boulder.nist.gov/ILT2/ilset?set={sid}"
    req = urllib.request.Request(url, headers=UA)
    j = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    head = j.get("dhead") or []
    if not head or "melting" not in str(head).lower():
        return None
    vals = []
    for row in j.get("data") or []:
        try:
            vals.append(float(row[0][0]))
        except Exception:
            continue
    return vals or None


def main() -> None:
    rows = json.loads((DATA / "ilt1.json").read_text())["res"]
    mp = [r for r in rows if r[2] == PROP and r[0] and r[8]]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [r for r in mp if r[0] not in cache]
    print(f"{len(mp):,} '{PROP}' datasets; {len(cache):,} cached; {len(todo):,} to fetch",
          flush=True)

    for i, r in enumerate(todo, 1):
        sid = r[0]
        try:
            cache[sid] = {"name": r[8], "K": fetch_set(sid)}
        except urllib.error.HTTPError as e:
            cache[sid] = {"name": r[8], "K": None, "err": f"HTTP {e.code}"}
        except Exception as e:
            cache[sid] = {"name": r[8], "K": None, "err": type(e).__name__}
        if i % 50 == 0 or i == len(todo):
            got = sum(1 for v in cache.values() if v.get("K"))
            print(f"  {i}/{len(todo)}  resolved {got:,}", flush=True)
            CACHE.write_text(json.dumps(cache))
        time.sleep(0.4)

    CACHE.write_text(json.dumps(cache))
    byname: dict[str, list[float]] = {}
    for v in cache.values():
        if v.get("K"):
            byname.setdefault(v["name"], []).extend(v["K"])
    print(f"\n{len(byname):,} compounds with at least one measured melting temperature")
    (DATA / "ilt_mp_by_name.json").write_text(json.dumps(byname))
    print(f"-> {DATA/'ilt_mp_by_name.json'}")


if __name__ == "__main__":
    main()
