"""Evaluation for the ionic-liquid pipeline: splits, chemical plausibility, sliced W2.

Nothing here touches training. Every function is a measurement, and in
particular `sliced_w2` is reported alongside the flow, never fed back into it.
"""
from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import wasserstein_distance

RDLogger.DisableLog("rdApp.*")


# --------------------------------------------------------------------- splits
def canon(s: str) -> str | None:
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None


def make_splits(ils: list[dict], seed: int = 0, test_ion_frac: float = 0.10,
                val_frac: float = 0.10, test_frac: float = 0.10) -> dict:
    """Four disjoint index sets over the IL corpus, in corpus order.

    `test_ion` is carved out FIRST and by whole cation, so no cation in it
    appears anywhere in train/valid/test. That is the real generalization test:
    69% of cations in this corpus are singletons, so a cation-disjoint split is
    cheap, whereas an anion-disjoint one is not (374 anions, the commonest
    covering 913 pairs).

    The remainder is split randomly by pair. A random held-out pair shares its
    anion with training almost always, so `test` measures recombination of known
    ions -- a strictly easier question than `test_ion`.
    """
    rng = np.random.default_rng(seed)
    cats = [canon(p["cation"]) or p["cation"] for p in ils]

    by_cat: dict[str, list[int]] = {}
    for i, c in enumerate(cats):
        by_cat.setdefault(c, []).append(i)

    # carve whole cation groups until the target fraction of PAIRS is reached
    order = list(by_cat); rng.shuffle(order)
    want, test_ion = int(round(test_ion_frac * len(ils))), []
    for c in order:
        if len(test_ion) >= want:
            break
        test_ion += by_cat[c]
    test_ion = np.array(sorted(test_ion))

    rest = np.array([i for i in range(len(ils)) if i not in set(test_ion.tolist())])
    rng.shuffle(rest)
    n_val, n_test = int(round(val_frac * len(rest))), int(round(test_frac * len(rest)))
    splits = {"valid": np.sort(rest[:n_val]),
              "test": np.sort(rest[n_val:n_val + n_test]),
              "train": np.sort(rest[n_val + n_test:]),
              "test_ion": test_ion}

    tr_cats = {cats[i] for i in splits["train"]}
    leak = tr_cats & {cats[i] for i in splits["test_ion"]}
    assert not leak, f"cation leak into test_ion: {len(leak)}"
    return splits


def make_splits_extended(ils: list[dict], n_original: int, seed: int = 0, **kw) -> dict:
    """Splits for a corpus that was EXTENDED after the splits were first drawn.

    The first `n_original` entries keep exactly the splits they had, so valid,
    test and test_ion are the same molecules as before; everything appended
    lands in train. That is what makes "more training data" measurable -- if the
    evaluation sets moved too, the comparison would confound two changes.

    scripts/build_il_merged.py guarantees the appended pairs contain no
    test_ion cation and no held-out pair, so the splits stay honest.
    """
    sp = make_splits(ils[:n_original], seed=seed, **kw)
    extra = np.arange(n_original, len(ils))
    sp["train"] = np.sort(np.concatenate([sp["train"], extra])) if len(extra) else sp["train"]
    cats = [canon(p["cation"]) or p["cation"] for p in ils]
    leak = {cats[i] for i in sp["train"]} & {cats[i] for i in sp["test_ion"]}
    assert not leak, f"cation leak into test_ion after extension: {len(leak)}"
    return sp


def ion_sets(ils: list[dict], idx) -> tuple[set, set, set]:
    """(pairs, cations, anions) present in a split, canonicalized."""
    P, C, A = set(), set(), set()
    for i in idx:
        c, a = canon(ils[i]["cation"]), canon(ils[i]["anion"])
        if c and a:
            P.add(f"{c}.{a}"); C.add(c); A.add(a)
    return P, C, A


# ------------------------------------------------------------------ chemistry
# Families that cover the IL literature, ordered specific -> generic because
# `_family` returns the first match. Patterns are LISTS, not comma-joined
# strings: SMARTS uses `,` as its own atom-level OR, so splitting on it turns
# `[F-,Cl-,Br-,I-]` into four invalid patterns that silently match nothing.
#
# Calibrated against the real corpus. If real ILs do not score near 1.0 on
# these, the pattern set is wrong, not the model.
CATION_SMARTS = {
    "imidazolium":   ["[n+]1ccnc1"],
    "pyridinium":    ["[n+]1ccccc1"],
    "pyrrolidinium": ["[N+]1CCCC1"],
    "piperidinium":  ["[N+]1CCCCC1"],
    "morpholinium":  ["[N+]1CCOCC1"],
    "guanidinium":   ["[NX3][CX3]=[NX3+]"],
    "phosphonium":   ["[PX4+]"],
    "sulfonium":     ["[SX3+]"],
    "ammonium":      ["[NX4+]"],
    "protic_N":      ["[NX3;H1,H2,H3;+]"],
    "other_cation":  ["[+]"],
}
ANION_SMARTS = {
    "metalate":      ["[Al-]", "[Ga-]", "[Fe-]", "[Zn-]", "[In-]", "[Sn-]",
                      "[Sb-]", "[As-]", "[Bi-]", "[Ti-]", "[Nb-]", "[Ta-]", "[Re-]"],
    "sulfonylimide": ["[N-](S(=O)=O)S(=O)=O"],
    "methide":       ["[C-](S(=O)=O)(S(=O)=O)S(=O)=O"],
    "dicyanamide":   ["[N-](C#N)C#N"],
    "tricyanomethanide": ["[#6-](C#N)(C#N)C#N"],
    "thiocyanate":   ["[S-]C#N", "[N-]=C=S"],
    "nitrate":       ["[NX3](=O)(=O)[O-]", "[N+](=O)([O-])[O-]"],
    "sulfate":       ["[OX2][SX4](=O)(=O)[O-]"],
    "sulfonate":     ["[SX4](=O)(=O)[O-]"],
    "phosphate":     ["[PX4](=O)[O-]"],
    "carboxylate":   ["[CX3](=O)[O-]"],
    "diketonate":    ["[CX3](=O)[#6-][CX3](=O)"],
    "cyanocarbanion": ["[#6-]C#N"],
    "halide":        ["[F-]", "[Cl-]", "[Br-]", "[I-]"],
    "fluoroborate":  ["[B-]"],
    "fluorophosph":  ["[P-]"],
    "azolate":       ["[n-]"],
    "aryl_carbanion": ["[c-]"],
    "amide_N":       ["[N-]"],
    "alkoxide":      ["[O-]"],
    "carbanion":     ["[#6-]"],
    "other_anion":   ["[-]"],
}
_CAT_PAT = {k: [Chem.MolFromSmarts(x) for x in v] for k, v in CATION_SMARTS.items()}
_AN_PAT = {k: [Chem.MolFromSmarts(x) for x in v] for k, v in ANION_SMARTS.items()}
for _k, _v in list(_CAT_PAT.items()) + list(_AN_PAT.items()):
    assert all(x is not None for x in _v), f"bad SMARTS in {_k}"

# Catch-alls exist so the family HISTOGRAM is complete, but they must not count
# toward `known_*_family` -- a metric that everything matches cannot fail.
CATCHALL = {"other_cation", "other_anion"}


def _family(mol, pats: dict) -> str | None:
    for name, ps in pats.items():
        if any(p is not None and mol.HasSubstructMatch(p) for p in ps):
            return name
    return None


def repair_radicals(smi: str) -> str:
    """Move radical electrons on bracket atoms into implicit hydrogens.

    A decoder emitting `[N+1]` or `[P+1]` writes a SMILES bracket atom, and the
    bracket convention is that such an atom takes NO implicit hydrogen -- so RDKit
    reads a protonated amine with three heavy bonds as a nitrogen radical. SELFIES'
    own valence model says N+ takes four bonds and intends the fourth to be an H;
    the bracket syntax simply does not write it.

    This is therefore a notation repair, not a chemistry edit. Round-tripping real
    corpus molecules leaves their radical count unchanged (0.33% before and after),
    so correct outputs are untouched; it recovers 6.5pp on the SELFIES ceiling and
    1.4pp on the SMILES one.
    """
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    changed = False
    for a in m.GetAtoms():
        r = a.GetNumRadicalElectrons()
        if r:
            a.SetNumExplicitHs(a.GetNumExplicitHs() + r)
            a.SetNumRadicalElectrons(0)
            changed = True
    if not changed:
        return smi
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return smi
    return Chem.MolToSmiles(m)


def plausibility(smiles: list[str], n_req: int, allowed_elements: set[int],
                 ref_heavy: np.ndarray | None = None,
                 ref_mw: np.ndarray | None = None) -> dict:
    """Chemical plausibility of a batch of generated (or real) SMILES.

    Layered, so a failure localizes. Every rate is over `n_req` requested
    samples except the family rates, which are over well-formed ion pairs --
    otherwise they would just restate `net_charge_zero`.

    `charge_balanced` in the old scorer only checked that one fragment was
    positive and one negative, which passes things like [Ca2+].[Cl-]. The real
    corpus is 4,788/4,790 exactly (+1,-1) with net charge zero, so
    `net_charge_zero` is the criterion that matches the data.
    """
    n_req = max(n_req, 1)
    parses = two_frag = net0 = charge11 = elem_ok = no_rad = 0
    cat_fam = an_fam = 0
    pairs, heavy, mw = [], [], []
    cat_families, an_families = {}, {}

    for s in smiles:
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        parses += 1
        fr = s.split(".")
        if len(fr) != 2:
            continue
        ms = [Chem.MolFromSmiles(f) for f in fr]
        if any(x is None for x in ms):
            continue
        two_frag += 1

        q = [Chem.GetFormalCharge(x) for x in ms]
        if sum(q) != 0 or not (max(q) > 0 and min(q) < 0):
            continue
        net0 += 1
        if sorted(q) == [-1, 1]:
            charge11 += 1

        if all(a.GetAtomicNum() in allowed_elements for a in m.GetAtoms()):
            elem_ok += 1
        else:
            continue
        if any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):
            continue
        no_rad += 1

        cm, am = (ms[0], ms[1]) if q[0] > 0 else (ms[1], ms[0])
        cf, af = _family(cm, _CAT_PAT), _family(am, _AN_PAT)
        cat_fam += cf is not None and cf not in CATCHALL
        an_fam += af is not None and af not in CATCHALL
        cat_families[cf or "unknown"] = cat_families.get(cf or "unknown", 0) + 1
        an_families[af or "unknown"] = an_families.get(af or "unknown", 0) + 1
        pairs.append(f"{Chem.MolToSmiles(cm)}.{Chem.MolToSmiles(am)}")
        heavy.append(m.GetNumHeavyAtoms()); mw.append(Descriptors.MolWt(m))

    nw = max(len(pairs), 1)
    out = {
        "parses": parses / n_req,
        "two_fragment": two_frag / n_req,
        "net_charge_zero": net0 / n_req,
        "charge_pm1": charge11 / n_req,
        "elements_ok": elem_ok / n_req,
        "no_radicals": no_rad / n_req,
        "plausible": len(pairs) / n_req,          # the headline: all of the above
        "known_cation_family": cat_fam / nw,
        "known_anion_family": an_fam / nw,
        "n_plausible": len(pairs),
        "cation_families": dict(sorted(cat_families.items(), key=lambda kv: -kv[1])[:6]),
        "anion_families": dict(sorted(an_families.items(), key=lambda kv: -kv[1])[:6]),
    }
    if heavy:
        out["heavy_mean"] = float(np.mean(heavy)); out["mw_mean"] = float(np.mean(mw))
        if ref_heavy is not None and len(ref_heavy):
            out["heavy_w1"] = float(wasserstein_distance(heavy, ref_heavy))
        if ref_mw is not None and len(ref_mw):
            out["mw_w1"] = float(wasserstein_distance(mw, ref_mw))
    return out, pairs


def element_set(smiles: list[str]) -> set[int]:
    """Atomic numbers occurring in a reference corpus -- the whitelist, learned."""
    z = set()
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            z |= {a.GetAtomicNum() for a in m.GetAtoms()}
    return z


def heavy_and_mw(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    h, w = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            h.append(m.GetNumHeavyAtoms()); w.append(Descriptors.MolWt(m))
    return np.array(h), np.array(w)


def novelty(gen_pairs: list[str], train_P: set, train_C: set, train_A: set) -> dict:
    """Novelty is always measured against TRAIN, never the full corpus."""
    uniq = set(gen_pairs)
    if not gen_pairs:
        return {"uniqueness": 0.0, "novel_combination": 0.0,
                "novel_cation_rate": 0.0, "novel_anion_rate": 0.0, "n_unique": 0}
    nc = sum(p.split(".")[0] not in train_C for p in gen_pairs)
    na = sum(p.split(".")[1] not in train_A for p in gen_pairs)
    return {"uniqueness": len(uniq) / len(gen_pairs),
            "novel_combination": len(uniq - train_P) / max(len(uniq), 1),
            "novel_cation_rate": nc / len(gen_pairs),
            "novel_anion_rate": na / len(gen_pairs),
            "n_unique": len(uniq)}


def recovery(gen_pairs: list[str], target_C: set, target_A: set) -> dict:
    """Did unconditional sampling ever reach ions held out from training?

    This is the generative half of the generalization question. The arms measure
    whether the model can DECODE held-out chemistry; this measures whether the
    flow ever PUTS MASS there without being asked to.
    """
    gc = {p.split(".")[0] for p in gen_pairs}
    ga = {p.split(".")[1] for p in gen_pairs}
    return {"cation_recall": len(gc & target_C) / max(len(target_C), 1),
            "anion_recall": len(ga & target_A) / max(len(target_A), 1),
            "n_cation_hit": len(gc & target_C), "n_anion_hit": len(ga & target_A)}


# ------------------------------------------------------------------ sliced W2
def sliced_w2(a, b, n_proj: int = 128, n_quant: int = 256, seed: int = 0,
              standardize: bool = True) -> float:
    """Sliced Wasserstein-2 between two point clouds.

    Project both clouds onto random unit directions, take the 1-D W2 on each
    (a sorted-quantile comparison), average. Zero iff the distributions match
    (Cramer-Wold), so it is a genuine metric on distributions.

    In this pipeline it is the ONLY metric measured on z before the decoder
    touches it, which is what makes it the flow's own score: validity and
    uniqueness are joint measurements of flow and decoder, the ceiling arm
    isolates the decoder, and this isolates the flow.

    `standardize` puts both clouds in the REFERENCE cloud's units (mean/std of
    `b`). Without it the number inherits the z-cloud's scale, which shrinks
    systematically as d grows -- so unstandardized values cannot be compared
    across a latent-dimension sweep, only within one run's own arms.

    Two blind spots worth remembering. It is insensitive to mode dropping inside
    a matched envelope, and random projections lose discriminative power as d
    grows (most 1-D projections of high-dimensional data look Gaussian), so it
    understates discrepancy at large d in addition to the scale effect.
    """
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if standardize:
        mu, sd = b.mean(0, keepdims=True), b.std(0, keepdims=True)
        sd = np.clip(sd, 1e-6, None)
        a = (a - mu) / sd; b = (b - mu) / sd
    r = np.random.default_rng(seed)
    v = r.standard_normal((n_proj, a.shape[1]))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    qs = np.linspace(0, 1, n_quant)
    pa = np.quantile(a @ v.T, qs, axis=0)
    pb = np.quantile(b @ v.T, qs, axis=0)
    return float(np.sqrt(((pa - pb) ** 2).mean()))
