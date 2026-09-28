"""Test the melting-point oracle against an independent compilation.

The structure oracle was fitted to 1,588 Zenodo labels, and the flow was
conditioned on the same 1,588. That shared origin is the weak point: a spurious
correlation in those labels would be learned by both and would never show up in
any internal check. NIST ILThermo is a separate compilation by separate people,
so it breaks the circle two ways.

  OVERLAP    compounds both sources measured. Their disagreement is the
             irreducible label noise -- the floor below which no oracle MAE can
             be judged good or bad, and which nothing here has measured.

  EXTERNAL   compounds only ILThermo has. The oracle never trained on them, the
             autoencoder never reconstructed them, the flow never saw them.
             Oracle error here is the honest estimate of its accuracy on
             molecules outside the training corpus -- which is the closest
             available proxy for its accuracy on GENERATED molecules.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from fm import Standardizer
from fm.il_eval import make_splits
from eval_cfg_structure import featurize
from eval_cfg_mp import train_oracle

DATA = Path("data")


def pair_form(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fr = smi.split(".")
    if len(fr) != 2:
        return None
    ms = [Chem.MolFromSmiles(f) for f in fr]
    if any(x is None for x in ms):
        return None
    q = [Chem.GetFormalCharge(x) for x in ms]
    if sum(q) != 0 or not (max(q) > 0 and min(q) < 0):
        return None
    cat, an = (ms[0], ms[1]) if q[0] > 0 else (ms[1], ms[0])
    return f"{Chem.MolToSmiles(cat)}.{Chem.MolToSmiles(an)}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/il/results/oracle_external.json")
    a = p.parse_args()

    nist = json.loads((DATA / "ilt_mp_by_name.json").read_text())
    cache = json.loads((DATA / "il_cir_cache.json").read_text())
    ils = json.loads((DATA / "il_pairs_v2.json").read_text())
    sp = make_splits(ils, seed=0)
    zen = {}
    for q in ils:
        v = q.get("melting_point_C")
        if v not in (None, ""):
            zen[q["pair"]] = float(v)
    zen_all = {q["pair"] for q in ils}

    rows = []
    for name, ks in nist.items():
        smi = cache.get(name)
        if not smi:
            continue
        pr = pair_form(smi)
        if pr is None:
            continue
        rows.append({"name": name, "pair": pr, "nist_C": float(np.median(ks)) - 273.15})
    byp = {}
    for r in rows:
        byp.setdefault(r["pair"], []).append(r["nist_C"])
    pairs = {k: float(np.median(v)) for k, v in byp.items()}
    print(f"ILThermo compounds with a melting point and a resolved structure: {len(pairs):,}")

    overlap = {k: v for k, v in pairs.items() if k in zen}
    external = {k: v for k, v in pairs.items() if k not in zen_all}
    print(f"  overlap with Zenodo labels : {len(overlap):,}")
    print(f"  external (not in corpus)   : {len(external):,}")

    # ---- 1. label noise between two independent compilations
    d = np.array([pairs[k] - zen[k] for k in overlap])
    print(f"\nLABEL NOISE — NIST vs Zenodo on {len(d):,} shared compounds")
    print(f"  median difference {np.median(d):+.1f} C   MAE {np.abs(d).mean():.1f} C   "
          f"RMS {np.sqrt((d**2).mean()):.1f} C")
    for q in (50, 75, 90):
        print(f"  {q}th pct |difference| {np.percentile(np.abs(d), q):.1f} C")
    print(f"  fraction agreeing within 10 C: {(np.abs(d) <= 10).mean():.1%}   "
          f"within 25 C: {(np.abs(d) <= 25).mean():.1%}")

    # ---- 2. oracle trained on Zenodo train-labelled, tested externally
    cond = json.loads((Path("runs/il/_fm_cfg_mp_d16") / "cond.json").read_text())
    mu, sd = cond["mean"], cond["std"]

    def build(split):
        X, Y = [], []
        for i in sp[split]:
            v = ils[i].get("melting_point_C")
            if v in (None, ""):
                continue
            f = featurize(ils[i]["pair"])
            if f is not None:
                X.append(f); Y.append(float(v))
        return np.array(X), np.array(Y)

    Xtr, Ytr = build("train")
    Xv, Yv = build("valid"); Xt, Yt = build("test")
    Xho, Yho = np.concatenate([Xv, Xt]), np.concatenate([Yv, Yt])
    # Drop descriptors that are CONSTANT in training before standardizing. The
    # Standardizer clamps std to 1e-6, which is correct for latents but ruinous
    # for sparse counts: an element absent from every training molecule has zero
    # variance, so one external molecule containing it standardizes to ~1e6 and
    # the unbounded MLP returns millions of degrees. Constant columns carry no
    # information anyway. Surviving values are clipped for safety.
    keep = np.asarray(Xtr).std(axis=0) > 1e-8
    print(f"  descriptors kept: {int(keep.sum())}/{len(keep)} "
          f"({int((~keep).sum())} constant in training, dropped)")
    Xtr, Xho = Xtr[:, keep], Xho[:, keep]
    fs = Standardizer().fit(torch.tensor(Xtr).float())
    _tf = fs.transform
    fs.transform = lambda x: _tf(x).clamp(-20, 20)
    net, mae_in = train_oracle(fs.transform(torch.tensor(Xtr).float()),
                               torch.tensor((Ytr - mu) / sd).float(),
                               fs.transform(torch.tensor(Xho).float()),
                               torch.tensor((Yho - mu) / sd).float())
    Xe, Ye = [], []
    for k, v in external.items():
        f = featurize(k)
        if f is not None:
            Xe.append(f); Ye.append(v)
    Xe, Ye = np.array(Xe)[:, keep], np.array(Ye)
    with torch.no_grad():
        pe = net(fs.transform(torch.tensor(Xe).float())).squeeze(-1).numpy() * sd + mu
    mae_ex = float(np.abs(pe - Ye).mean())
    base_ex = float(np.abs(Ye - Ye.mean()).mean())
    print(f"\nORACLE GENERALIZATION")
    print(f"  in-corpus held-out (Zenodo valid+test, n={len(Yho)}): MAE {mae_in*sd:.1f} C")
    print(f"  EXTERNAL (ILThermo, not in corpus, n={len(Ye)}):      MAE {mae_ex:.1f} C   "
          f"(mean baseline {base_ex:.1f} C)")
    sl = float(np.polyfit(Ye, pe, 1)[0])
    r = float(np.corrcoef(Ye, pe)[0, 1])
    print(f"  external slope {sl:.3f}   R {r:.3f}   (1.0 = no shrinkage)")
    print(f"  median abs error {np.median(np.abs(pe - Ye)):.1f} C")
    print(f"  baseline, predict the TRAINING mean ({mu:.0f} C): "
          f"{np.abs(Ye - mu).mean():.1f} C")
    print(f"  baseline, predict the external set's own mean: {base_ex:.1f} C "
          f"(an oracle cannot know this)")
    print(f"  external true MP: median {np.median(Ye):.1f} C  "
          f"p10 {np.percentile(Ye,10):.0f}  p90 {np.percentile(Ye,90):.0f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"n_pairs": len(pairs), "n_overlap": len(overlap), "n_external": len(Ye),
         "label_noise_MAE_C": float(np.abs(d).mean()),
         "label_noise_RMS_C": float(np.sqrt((d**2).mean())),
         "label_noise_median_C": float(np.median(d)),
         "oracle_mae_incorpus_C": float(mae_in * sd),
         "oracle_mae_external_C": mae_ex, "oracle_baseline_external_C": base_ex,
         "oracle_slope_external": sl}, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
