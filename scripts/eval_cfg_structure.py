"""Does the conditioning survive the decoder?

eval_cfg_mp.py scored generated LATENTS with a regressor f(z) -> MP. That shows
the flow moved z to where ILs of the requested melting point live, but it cannot
show the decoded MOLECULE has that melting point -- the decoder could map a
well-placed latent onto something else entirely, and a latent-space oracle would
never notice.

So this builds a second oracle that never sees the latent space at all: RDKit
descriptors computed on the cation and anion separately, into a small MLP. It
shares nothing with the flow but the training labels -- not ChemBERTa, not
MLP_down, not the frozen z geometry -- so agreement between the two curves is
real evidence and disagreement localizes the failure to the decoder.

Both oracles score the SAME generated batch, so the comparison is paired and
carries no sampling noise between them.

Two honest limits. Only molecules that parse as a charge-balanced ion pair can
be featurized, so the structure curve is conditioned on passing plausibility --
the fraction scored is reported per cell. And the oracle's held-out MAE is
measured on REAL held-out ILs, which is an optimistic floor for its accuracy on
generated ones.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for train_oracle
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

from fm import CFGVelocity, Standardizer, sample
from fm.il_eval import make_splits
from fm.il_io import (decode, encode_all, load_ae, load_cfg_fm, load_cond, load_emb, load_ils,
                      load_tokenizer, write_json)
from eval_cfg_mp import train_oracle

ELEMS = ["C", "N", "O", "F", "S", "P", "B", "Cl", "Br", "I"]


def frag_desc(m) -> list[float]:
    cnt = {e: 0 for e in ELEMS}
    for at in m.GetAtoms():
        s = at.GetSymbol()
        if s in cnt:
            cnt[s] += 1
    return [
        Descriptors.MolWt(m), float(m.GetNumHeavyAtoms()),
        float(rdMolDescriptors.CalcNumRotatableBonds(m)),
        rdMolDescriptors.CalcTPSA(m), Descriptors.MolLogP(m),
        float(rdMolDescriptors.CalcNumRings(m)),
        float(rdMolDescriptors.CalcNumAromaticRings(m)),
        rdMolDescriptors.CalcFractionCSP3(m),
        float(rdMolDescriptors.CalcNumHBD(m)), float(rdMolDescriptors.CalcNumHBA(m)),
        float(Chem.GetFormalCharge(m)),
    ] + [float(cnt[e]) for e in ELEMS]


def featurize(smi: str):
    """cation descriptors ++ anion descriptors ++ 3 pair terms, or None."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
    if len(fr) != 2:
        return None
    q = [Chem.GetFormalCharge(f) for f in fr]
    if sum(q) != 0 or not (max(q) > 0 and min(q) < 0):
        return None
    cat, an = (fr[0], fr[1]) if q[0] > 0 else (fr[1], fr[0])
    try:
        dc, da = frag_desc(cat), frag_desc(an)
    except Exception:
        return None
    mw_c, mw_a = dc[0], da[0]
    extra = [mw_c + mw_a, mw_c / max(mw_a, 1e-6),
             dc[1] / max(da[1], 1e-6)]
    v = dc + da + extra
    return None if not all(np.isfinite(v)) else v


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="cfg_mp_d16")
    p.add_argument("--ae", default="runs/il/ae/ae_enum_d16.pt")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--targets", default="0,40,80,120,160,200")
    p.add_argument("--weights", default="0,1,2")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--out", default="runs/il/results/cfg_mp_structure.json")
    a = p.parse_args()

    dev = torch.device("cpu"); torch.manual_seed(0)
    tok = load_tokenizer()

    ils = load_ils()
    sp = make_splits(ils, seed=0)
    mu, sd, raw = load_cond(a.tag, ils)

    # ---- structure oracle: real labelled ILs -> descriptors -> MP
    def build(split):
        X, Y = [], []
        for i in sp[split]:
            if not np.isfinite(raw[i]):
                continue
            f = featurize(ils[i]["pair"])
            if f is not None:
                X.append(f); Y.append(raw[i])
        return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)

    Xtr, Ytr = build("train")
    Xv, Yv = build("valid"); Xt, Yt = build("test")
    Xho, Yho = np.concatenate([Xv, Xt]), np.concatenate([Yv, Yt])
    fs = Standardizer().fit(torch.tensor(Xtr).float())
    Ztr_s = fs.transform(torch.tensor(Xtr).float())
    Zho_s = fs.transform(torch.tensor(Xho).float())
    print(f"structure oracle: {len(Xtr):,} train / {len(Xho):,} held-out labelled ILs, "
          f"{Xtr.shape[1]} descriptors")
    s_oracle, s_mae = train_oracle(Ztr_s, torch.tensor((Ytr - mu) / sd).float(),
                                   Zho_s, torch.tensor((Yho - mu) / sd).float())
    s_mae_c = s_mae * sd
    s_base = float(np.abs(Yho - Yho.mean()).mean())
    print(f"  held-out MAE {s_mae_c:.1f} C   (mean baseline {s_base:.1f} C)")

    # ---- latent oracle, same as eval_cfg_mp.py
    E = load_emb()
    ae = load_ae(a.ae, a.latent_dim, tok, 80, dev)
    Zall = encode_all(ae, E)
    sc = Standardizer().fit(Zall[torch.from_numpy(sp["train"]).long()])
    Zs = sc.transform(Zall)

    def lat_xy(name):
        idx = sp[name]; sel = idx[np.isfinite(raw[idx])]
        return Zs[torch.from_numpy(sel).long()], torch.tensor(raw[sel]).float()
    Ltr, ltr = lat_xy("train")
    lv, yv2 = lat_xy("valid"); lt, yt2 = lat_xy("test")
    Lho, lho = torch.cat([lv, lt]), torch.cat([yv2, yt2])
    l_oracle, l_mae = train_oracle(Ltr, (ltr - mu) / sd, Lho, (lho - mu) / sd)
    print(f"latent oracle:    held-out MAE {l_mae*sd:.1f} C\n")

    fm = load_cfg_fm(a.tag, a.latent_dim, dev)

    targets = [float(x) for x in a.targets.split(",")]
    out = {"structure_mae_C": s_mae_c, "structure_baseline_C": s_base,
           "latent_mae_C": l_mae * sd, "n": a.n, "curve": {}}

    for w in [float(x) for x in a.weights.split(",")]:
        g = CFGVelocity(fm, w=w); lat, struct, frac, wild = [], [], [], []
        for T in targets:
            yv = torch.tensor([[(T - mu) / sd, 1.0]]).repeat(a.n, 1)
            with torch.no_grad():
                z = sample(g, n=a.n, shape=(a.latent_dim,), y=yv,
                           n_steps=a.ode_steps, method="heun", device=dev)
                lat.append(float(np.median(l_oracle(z).squeeze(-1).numpy())) * sd + mu)
                smis = decode(ae, sc.inverse(z), tok, 80, repair=False)
                F = [f for f in (featurize(s) for s in smis) if f is not None]
                frac.append(len(F) / max(len(smis), 1))
                if F:
                    Xf = fs.transform(torch.tensor(np.array(F)).float())
                    pr = s_oracle(Xf).squeeze(-1).numpy() * sd + mu
                    # MEDIAN, not mean: the oracle is an unbounded MLP and a
                    # malformed generated molecule can land far outside the
                    # descriptor range it was fitted on, producing predictions in
                    # the thousands of degrees. One such sample ruins a mean of
                    # 1,000; the median is unaffected. The share of predictions
                    # outside a physically sensible window is reported so the
                    # problem stays visible rather than being silently absorbed.
                    struct.append(float(np.median(pr)))
                    wild.append(float(np.mean((pr < -150) | (pr > 500))))
                else:
                    struct.append(float("nan")); wild.append(float("nan"))
            print(f"  w={w:.1f} target {T:5.0f}  latent {lat[-1]:6.1f}  "
                  f"structure {struct[-1]:6.1f}  scored {frac[-1]:5.1%}  "
                  f"wild {wild[-1]:4.1%}", flush=True)
        T_ = np.array(targets)
        sl_l = float(np.polyfit(T_, np.array(lat), 1)[0])
        ok = np.isfinite(struct)
        sl_s = float(np.polyfit(T_[ok], np.array(struct)[ok], 1)[0]) if ok.sum() > 1 else float("nan")
        r_s = float(np.corrcoef(T_[ok], np.array(struct)[ok])[0, 1]) if ok.sum() > 1 else float("nan")
        out["curve"][str(w)] = {"targets": targets, "latent": lat, "structure": struct,
                                "frac_scored": frac, "frac_wild": wild, "slope_latent": sl_l,
                                "slope_structure": sl_s, "r_structure": r_s}
        print(f"  -> w={w:.1f}  slope latent {sl_l:.3f}   slope structure {sl_s:.3f}  "
              f"(R {r_s:.3f})\n", flush=True)

    write_json(a.out, out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
