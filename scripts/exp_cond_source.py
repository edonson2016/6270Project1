"""Property control through the SOURCE instead of through guidance.

Classifier-free guidance (6.14) puts the melting point into the velocity field:
c = time_embed(t) + cond_embed(y), then extrapolate at sampling. It works, but it
costs ~0.07 of sw2_test, needs a guidance weight, and the label competes with time
for the same 384 channels.

This tries the other place a conditional generative model can carry a condition:
the SOURCE. Bin the train-labelled melting points, fit one full-covariance Gaussian
per band, and start the flow inside the band you want. The source is therefore a
Gaussian MIXTURE whose components ARE melting-point modes -- and to sample at a
target melting point you simply pick the component. No guidance, no conditioning
channel, 1x NFE.

  method       : PriorGrad (arXiv 2106.06406) is the published form of a
                 data-dependent prior built from conditioning information.
  the OT fix   : minibatch OT ignores the condition and skews the prior
                 conditionally (C2OT, arXiv 2503.10636). Here the fix is exact
                 rather than a soft penalty: x0 is drawn from the band of its own
                 x1, so the assignment is solved BLOCK-WISE within bands and can
                 never pair a low-melting source with a high-melting target.

Judged by the same oracle-free band-matching test as 6.14 -- generate at a band,
check the generated latent cloud sits closest to REAL held-out ILs measured in
that band -- so it is directly comparable and needs no property predictor.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm import FlowMatching, Standardizer, VelocityMLP, sample
from fm.model import ot_pair
from fm.paths import CondOTPath
from fm.il_eval import make_splits, plausibility, novelty, sliced_w2
from fm.il_io import (RESULTS, decode, encode_all, load_ae, load_emb, load_ils, load_tokenizer,
                      train_reference, write_json)

EDGES = [-100, 20, 60, 100, 140, 180, 400]


def block_ot(x0, x1, g):
    """C2OT, exactly: permute x0 within each conditioning group only."""
    out = x0.clone()
    for b in g.unique():
        m = (g == b).nonzero(as_tuple=True)[0]
        if len(m) > 1:
            out[m] = ot_pair(x0[m], x1[m])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ae", default="runs/il/ae/ae_sf_d16.pt")
    p.add_argument("--vocab", default="data/selfies_vocab.json")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--sigma-min", type=float, default=0.10)
    p.add_argument("--coupling", default="c2ot", choices=["c2ot", "ot", "independent"])
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--cond-channel", action="store_true",
                   help="ALSO put the band in the velocity field (cond_dim=2), not only in the "
                        "source. The 3/6 band-match from source-only conditioning says the field "
                        "cannot keep band identity where bands overlap, because v(x,t) has no "
                        "label to read; this gives it one.")
    p.add_argument("--guide-w", type=float, default=1.0)
    p.add_argument("--source-dropout", type=float, default=0.0,
                   help="probability of drawing x0 from the GLOBAL source instead of the band's. "
                        "A conditional source explains the condition away, so the conditioning "
                        "channel never learns to use it (band-match 3/6 with the channel, against "
                        "6/6 for CFG from an N(0,I) start). This is CFG's own null-token dropout "
                        "applied to the PRIOR: withhold the informative source sometimes, and the "
                        "channel has to carry the band or the model loses it.")
    p.add_argument("--sample-mode", default="prior", choices=["prior", "band0"],
                   help="'prior' draws the band from its empirical training frequency, which is "
                        "true UNCONDITIONAL generation from the mixture and the only number "
                        "comparable to an unconditional baseline. 'band0' reproduces the earlier "
                        "buggy slice that reported band 0 alone.")
    p.add_argument("--tag", default="condsrc")
    a = p.parse_args()

    dev = torch.device("cpu"); torch.manual_seed(0)
    tok = load_tokenizer("selfies", a.vocab)
    ae = load_ae(a.ae, a.latent_dim, tok, 96, dev)

    ils = load_ils()
    sp = make_splits(ils, seed=0)
    Z = encode_all(ae, load_emb())
    sc = Standardizer().fit(Z[torch.from_numpy(sp["train"]).long()])
    Zs = sc.transform(Z)
    mp = np.array([float(q["melting_point_C"]) if q.get("melting_point_C") not in (None, "")
                   else np.nan for q in ils])

    def band_of(v):
        return int(np.clip(np.digitize(v, EDGES) - 1, 0, len(EDGES) - 2))

    tr = np.asarray(sp["train"])
    B = len(EDGES) - 1
    # band label per train row; unlabelled rows get a band sampled from the label prior
    lab = np.isfinite(mp[tr])
    g = np.array([band_of(mp[i]) if np.isfinite(mp[i]) else -1 for i in tr])
    prior = np.bincount(g[g >= 0], minlength=B) / lab.sum()
    rng = np.random.default_rng(0)
    g[g < 0] = rng.choice(B, size=(g < 0).sum(), p=prior)
    X1 = Zs[torch.from_numpy(tr).long()]
    G = torch.from_numpy(g).long()

    # ---- the source: one full-covariance Gaussian per melting-point band
    mus, Ls, ns = [], [], []
    for b in range(B):
        m = (G == b)
        x = X1[m].double()
        ns.append(int(m.sum()))
        mus.append(x.mean(0))
        c = torch.cov(x.T) + 1e-4 * torch.eye(a.latent_dim, dtype=torch.float64)
        Ls.append(torch.linalg.cholesky(c))
    xg = X1.double()
    mu_g = xg.mean(0)
    L_g = torch.linalg.cholesky(torch.cov(xg.T) + 1e-4 * torch.eye(a.latent_dim, dtype=torch.float64))
    print(f"source: {B} band components, sizes {ns}  (labelled {int(lab.sum())} of {len(tr)}; "
          f"unlabelled assigned by the label prior)", flush=True)

    def draw(bands: torch.Tensor, dropout: float = 0.0) -> torch.Tensor:
        z = torch.randn(len(bands), a.latent_dim, dtype=torch.float64)
        drop = (torch.rand(len(bands)) < dropout) if dropout > 0 else torch.zeros(len(bands), dtype=torch.bool)
        out = torch.stack([(mu_g + L_g @ z[i]) if drop[i]
                           else (mus[int(b)] + Ls[int(b)] @ z[i])
                           for i, b in enumerate(bands)])
        return out.float(), drop
    def draw0(bands, dropout=0.0):
        return draw(bands, dropout)[0]

    # ---- train
    CD = 2 if a.cond_channel else 0
    def yvec(bands):
        return torch.stack([(bands.float() / (B - 1)) * 2 - 1, torch.ones(len(bands))], 1)
    net = VelocityMLP(a.latent_dim, cond_dim=CD, width=384, depth=4)
    fm = FlowMatching(net, path=CondOTPath(sigma_min=a.sigma_min))
    opt = torch.optim.AdamW(fm.parameters(), lr=1e-3, weight_decay=0.0)
    import math
    def lr_at(s):
        if s < 500: return 1e-3 * (s + 1) / 500
        q = (s - 500) / max(1, a.steps - 500)
        return 1e-3 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(q, 1.0))))
    ema = {k: v.detach().clone() for k, v in fm.state_dict().items()}
    dl = DataLoader(TensorDataset(X1, G), batch_size=128, shuffle=True, drop_last=True)
    it = iter(dl); t0 = time.time()
    for s in range(a.steps):
        try: xb, gb = next(it)
        except StopIteration: it = iter(dl); xb, gb = next(it)
        x0, drop = draw(gb, a.source_dropout)
        # dropped rows are not band-aligned, so they form their own OT block
        gb_ot = torch.where(drop, torch.full_like(gb, B), gb)
        if a.coupling == "c2ot": x0 = block_ot(x0, xb, gb_ot)
        elif a.coupling == "ot": x0 = ot_pair(x0, xb)
        for pg in opt.param_groups: pg["lr"] = lr_at(s)
        loss = fm.loss(xb, y=yvec(gb) if CD else None, x0=x0)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for k, v in fm.state_dict().items(): ema[k].lerp_(v, 0.001)
        if (s + 1) % 2000 == 0:
            print(f"  step {s+1}/{a.steps}  loss {float(loss):.4f}  {time.time()-t0:.0f}s", flush=True)
    fm.load_state_dict(ema); fm.eval()

    # ---- evaluate: oracle-free band matching + quality
    ho = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        idx = [i for k in ("valid", "test") for i in sp[k]
               if np.isfinite(mp[i]) and lo <= mp[i] < hi]
        ho.append(Zs[torch.from_numpy(np.array(idx)).long()].numpy())
    M = np.zeros((B, B))
    gens = []
    for b in range(B):
        with torch.no_grad():
            bb = torch.full((a.n,), b)
            mdl = fm
            if CD and a.guide_w != 1.0:
                from fm.guidance import CFGVelocity
                mdl = CFGVelocity(fm, w=a.guide_w)
            z = sample(mdl, x0=draw0(bb), y=yvec(bb) if CD else None,
                       n_steps=a.ode_steps, method="heun", device=dev)
        gens.append(z)
        for j in range(B): M[b, j] = sliced_w2(z.numpy(), ho[j])
    diag = sum(1 for j in range(B) if int(np.argmin(M[:, j])) == j)
    print(f"\nBAND MATCH (rows=requested band, cols=real held-out band)")
    lbl = [f"{lo}-{hi}" for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    print("        " + "".join(f"{l:>10}" for l in lbl))
    best = np.argmin(M, axis=0)
    for i in range(B):
        print(f"{lbl[i]:>8}" + "".join(f"{M[i,j]:9.3f}" + ("*" if best[j] == i else " ")
                                      for j in range(B)))
    print(f"column minima on diagonal: {diag}/{B}   diag mean {np.mean(np.diag(M)):.4f}")

    trainP, trainC, trainA, ELEM, RH, RM = train_reference(ils, sp["train"])
    # per-band quality: decodability varies strongly with melting point, so a pooled
    # number is only meaningful once bands are drawn at their training frequency
    per_band = []
    for b in range(B):
        sm = decode(ae, sc.inverse(gens[b][:500]), tok, 96)
        mb, _ = plausibility(sm, len(sm), ELEM, RH, RM)
        per_band.append(mb["plausible"])
    print("\nper-band plausibility: " + "  ".join(
        f"{lbl[b]}={per_band[b]:.3f}" for b in range(B)))

    if a.sample_mode == "prior":
        pick = rng.choice(B, size=a.n, p=prior)
        zc = torch.stack([gens[int(b)][i] for i, b in enumerate(pick)])
        print(f"pooled at training band frequency {np.round(prior,3).tolist()}")
    else:
        zc = torch.cat(gens)[: a.n]
        print("pooled from band 0 only (the earlier buggy slice)")
    smis = decode(ae, sc.inverse(zc), tok, 96)
    m, pairs = plausibility(smis, len(smis), ELEM, RH, RM)
    m |= novelty(pairs, trainP, trainC, trainA)
    print(f"\nquality ({a.sample_mode}-pooled, n={len(smis)}): plaus {m['plausible']:.4f}  "
          f"uniq {m['uniqueness']:.4f}  novcat {m['novel_cation_rate']:.4f}  NFE {2*a.ode_steps}")
    write_json(RESULTS / f"{a.tag}.json",
        {"coupling": a.coupling, "sigma_min": a.sigma_min,
         "cond_channel": bool(a.cond_channel), "guide_w": a.guide_w,
         "source_dropout": a.source_dropout, "sample_mode": a.sample_mode,
         "per_band_plausible": per_band, "matrix": M.tolist(),
         "diag": diag, "diag_mean": float(np.mean(np.diag(M))),
         "plausible": m["plausible"], "uniqueness": m["uniqueness"],
         "novel_cation_rate": m["novel_cation_rate"], "band_sizes": ns})
    print(f"-> runs/il/results/{a.tag}.json")


if __name__ == "__main__":
    main()
