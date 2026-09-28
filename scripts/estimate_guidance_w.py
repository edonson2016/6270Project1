"""Predict whether autoguidance can help, and at what w, WITHOUT sampling.

Autoguidance extrapolates away from a degraded model:

    v~_w = (1+w)*v1 - w*v0

Write v1 = v* + e1 and v0 = v* + e0, and let D = v1 - v0 = e1 - e0. Then

    v~_w - v* = e1 + w*D
    E||v~_w - v*||^2 = E||e1||^2 + 2w*E<e1,D> + w^2*E||D||^2

so guidance reduces error iff E<e1,D> < 0, with optimum

    w* = -E<e1,D> / E||D||^2

E<e1,D> is directly estimable even though v* is unknown: the flow matching
target u_t = x1 - x0 satisfies E[u_t | x_t,t] = v*, and D is a function of
(x_t,t), so

    E<v1(x_t,t) - u_t, D(x_t,t)>  =  E<e1, D>

exactly, over held-out x1. Two forward passes per draw, no generation.

This is a gate, not a diagnostic: a POSITIVE numerator means autoguidance
cannot help at any w > 0 with that guiding model, and the expensive sampling
sweep should not be run. Binning by t additionally gives w*(t), which is the
principled form of the finding that guidance should be limited to an interval
of noise levels.

    scripts/estimate_guidance_w.py enum_fm_d16 ag_w05 ag_w10 ag_w25 ...
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm import FlowMatching, Standardizer, VelocityMLP
from fm.il_eval import make_splits, make_splits_extended
from fm.il_io import encode_all, load_ae, load_emb, load_ils, load_tokenizer, write_json


def load_field(tag: str, D: int, width: int = 384) -> FlowMatching:
    fm = FlowMatching(VelocityMLP(D, width=width, depth=4))
    sd = torch.load(f"runs/il/_fm_{tag}/last.pt", map_location="cpu", weights_only=False)
    fm.load_state_dict(sd["ema"]); fm.eval()
    return fm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("strong", help="tag of v1")
    p.add_argument("weak", nargs="+", help="tags of candidate v0")
    p.add_argument("--ae", default="runs/il/ae/ae_enum_d16.pt")
    p.add_argument("--corpus", default="il_pairs_v2")
    p.add_argument("--n-original", type=int, default=0)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--draws", type=int, default=64, help="t/x0 draws per held-out latent")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--split", default="valid")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/il/results/guidance_w.json")
    a = p.parse_args()

    dev = torch.device("cpu"); torch.manual_seed(a.seed)
    tok = load_tokenizer()
    m = load_ae(a.ae, a.latent_dim, tok, 80, dev)

    ils = load_ils(a.corpus)
    esuf = "" if a.corpus == "il_pairs_v2" else "_" + a.corpus.rsplit("_", 1)[-1]
    E = load_emb(f"il_emb_ils{esuf}.npy")
    sp = (make_splits_extended(ils, a.n_original) if a.n_original else make_splits(ils))
    Z = encode_all(m, E)

    # the frame every field was trained in: fit on the FULL train cloud
    sc = Standardizer().fit(Z[torch.from_numpy(sp["train"]).long()])
    X1 = sc.transform(Z[torch.from_numpy(sp[a.split]).long()])
    print(f"held-out {a.split}: {tuple(X1.shape)}   draws/latent {a.draws}"
          f"   total {len(X1)*a.draws:,}\n", flush=True)

    v1 = load_field(a.strong, a.latent_dim)

    g = torch.Generator().manual_seed(a.seed)
    X1r = X1.repeat_interleave(a.draws, 0)
    X0 = torch.randn(X1r.shape, generator=g)
    T = torch.rand(len(X1r), generator=g)
    Xt = (1 - T[:, None]) * X0 + T[:, None] * X1r
    U = X1r - X0

    with torch.no_grad():
        V1 = torch.cat([v1(Xt[i:i+4096], T[i:i+4096]) for i in range(0, len(Xt), 4096)])
    resid = V1 - U                                    # unbiased for e1 in any inner product
    fm_val = float(resid.pow(2).sum(1).mean())        # ||e1||^2 + irreducible noise

    out = {"strong": a.strong, "split": a.split, "n_draws": int(len(Xt)),
           "E_resid_sq": fm_val, "candidates": {}}
    print(f"{'candidate':12} {'E<e1,D>':>10} {'E|D|^2':>9} {'w*':>7} "
          f"{'pred.red.':>10} {'% of E|v1-u|^2':>15}   verdict")
    print("-" * 82)

    for tag in a.weak:
        v0 = load_field(tag, a.latent_dim)
        with torch.no_grad():
            V0 = torch.cat([v0(Xt[i:i+4096], T[i:i+4096]) for i in range(0, len(Xt), 4096)])
        Dv = V1 - V0
        num_i = (resid * Dv).sum(1)                   # <v1-u_t, D>  per draw
        den_i = Dv.pow(2).sum(1)                      # ||D||^2      per draw
        num, den = float(num_i.mean()), float(den_i.mean())
        w_star = -num / den if den > 0 else float("nan")
        red = num * num / den if den > 0 else 0.0     # predicted MSE reduction at w*
        ok = num < 0
        # standard error on the numerator, so a near-zero verdict is not over-read
        se = float(num_i.std() / np.sqrt(len(num_i)))
        verdict = ("HELPS" if ok else "CANNOT HELP") + ("" if abs(num) > 2 * se else "  (within 2 SE of 0)")
        print(f"{tag:12} {num:10.4f} {den:9.4f} {w_star:7.3f} {red:10.4f} "
              f"{100*red/max(fm_val,1e-12):14.2f}%   {verdict}")

        # w*(t): where in the path the guidance signal actually lives
        edges = np.linspace(0, 1, a.bins + 1)
        idx = np.clip(np.digitize(T.numpy(), edges) - 1, 0, a.bins - 1)
        per_t = []
        for b in range(a.bins):
            msk = idx == b
            if msk.sum() < 32:
                per_t.append(None); continue
            nb, db = float(num_i[msk].mean()), float(den_i[msk].mean())
            per_t.append({"t_lo": float(edges[b]), "t_hi": float(edges[b+1]),
                          "num": nb, "den": db, "w_star": -nb/db if db > 0 else None,
                          "reduction": nb*nb/db if db > 0 else 0.0})
        out["candidates"][tag] = {"num": num, "num_se": se, "den": den,
                                  "w_star": w_star, "predicted_reduction": red,
                                  "helps": bool(ok), "per_t": per_t}

    best = [k for k, v in out["candidates"].items() if v["helps"]]
    best.sort(key=lambda k: -out["candidates"][k]["predicted_reduction"])
    out["ranked"] = best
    print()
    if best:
        b = out["candidates"][best[0]]
        print(f"BEST: {best[0]}   w* = {b['w_star']:.3f}   "
              f"predicted reduction {b['predicted_reduction']:.4f}")
        print("  w*(t) by decile: " + "  ".join(
            "—" if q is None else f"{q['t_lo']:.1f}:{q['w_star']:+.2f}" for q in b["per_t"]))
    else:
        print("NO CANDIDATE HELPS — every numerator is positive. Autoguidance with these")
        print("guiding models cannot reduce error at any w > 0. Do not run the sweep.")

    write_json(a.out, out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
