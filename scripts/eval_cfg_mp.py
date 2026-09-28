"""Does conditioning on melting point actually steer the flow?

The headline is a CONTROL CURVE: sweep the requested melting point across the
training range and measure what the generated latents actually come out at. A
slope near 1 means the label controls generation; a slope near 0 means the
network learned to ignore it. Everything else -- plausibility, uniqueness -- is
the price paid for whatever control is achieved.

The oracle is a separate regressor f(z) -> MP trained on the same 1,588
train-labelled latents. That is NOT circular: classifier-free guidance conditions
the flow directly on the label during training and never consults a regressor,
so the regressor is an independent read on where a generated latent landed. What
it cannot do is tell you the decoded MOLECULE has that melting point -- it only
says the flow put the latent where ILs of that melting point live. Its own
held-out MAE is printed first, as the floor below which no control claim means
anything: against a label sd of ~78 C, an MAE near 40 C is usable and an MAE
near 70 C means the oracle is barely better than guessing the mean.

Sampling latents is cheap and decoding is not, so the control curve runs over
the full (target x w) grid on latents alone, and only a few cells are decoded
for the quality/diversity cost.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm import CFGVelocity, Standardizer, guided_nfe, sample
from fm.il_eval import make_splits, plausibility, novelty
from fm.il_io import (decode, encode_all, load_ae, load_cfg_fm, load_cond, load_emb, load_ils,
                      load_tokenizer, train_reference, write_json)


def train_oracle(Ztr, ytr, Zva, yva, epochs=400, seed=0):
    """Small MLP regressor on standardized latents. Returns (model, held-out MAE)."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Ztr.shape[1], 64), nn.SiLU(),
                        nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_sd = float("inf"), None
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Ztr))
        for i in range(0, len(Ztr), 128):
            b = perm[i:i+128]
            loss = nn.functional.mse_loss(net(Ztr[b]).squeeze(-1), ytr[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            mae = float((net(Zva).squeeze(-1) - yva).abs().mean())
        if mae < best:
            best, best_sd = mae, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_sd); net.eval()
    return net, best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="cfg_mp_d16")
    p.add_argument("--ae", default="runs/il/ae/ae_enum_d16.pt")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--targets", default="0,40,80,120,160,200", help="requested MP, degC")
    p.add_argument("--weights", default="0,1,2,3,5", help="CFG guidance weights")
    p.add_argument("--n-curve", type=int, default=2000, help="latents per cell (no decode)")
    p.add_argument("--decode-w", default="0,1,3", help="which w to also decode")
    p.add_argument("--decode-target", type=float, default=80.0)
    p.add_argument("--n-decode", type=int, default=1000)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--out", default="runs/il/results/cfg_mp.json")
    a = p.parse_args()

    dev = torch.device("cpu"); torch.manual_seed(0)
    tok = load_tokenizer()
    ae = load_ae(a.ae, a.latent_dim, tok, 80, dev)

    ils = load_ils()
    E = load_emb()
    sp = make_splits(ils, seed=0)
    Zall = encode_all(ae, E)
    sc = Standardizer().fit(Zall[torch.from_numpy(sp["train"]).long()])
    Zs = sc.transform(Zall)

    mu, sd, raw = load_cond(a.tag, ils)

    def split_xy(name):
        idx = sp[name]; lab = np.isfinite(raw[idx]); sel = idx[lab]
        return Zs[torch.from_numpy(sel).long()], torch.tensor(raw[sel]).float()

    Ztr, ytr = split_xy("train")
    Zva, yva = split_xy("valid"); Zte, yte = split_xy("test")
    Zho = torch.cat([Zva, Zte]); yho = torch.cat([yva, yte])
    print(f"oracle training set {len(Ztr):,} labelled train latents; held-out {len(Zho):,}")
    oracle, mae = train_oracle(Ztr, (ytr - mu) / sd, Zho, (yho - mu) / sd)
    mae_c = mae * sd
    base = float((yho - yho.mean()).abs().mean())
    print(f"ORACLE held-out MAE {mae_c:.1f} C   (predict-the-mean baseline {base:.1f} C, "
          f"label sd {sd:.1f} C)")
    if mae_c > 0.85 * base:
        print("  !! oracle is barely better than the mean -- control claims below are weak")
    print()

    fm = load_cfg_fm(a.tag, a.latent_dim, dev)

    targets = [float(x) for x in a.targets.split(",")]
    weights = [float(x) for x in a.weights.split(",")]
    out = {"tag": a.tag, "oracle_mae_C": mae_c, "oracle_baseline_C": base,
           "label_mean": mu, "label_sd": sd, "curve": {}, "decoded": {}}

    print("CONTROL CURVE — requested MP vs achieved (oracle read on generated latents)")
    hdr = "  w   " + "".join(f"{t:>9.0f}" for t in targets) + "     slope      R"
    print(hdr); print("-" * len(hdr))
    for w in weights:
        g = CFGVelocity(fm, w=w)
        ach = []
        for T in targets:
            yv = torch.tensor([[(T - mu) / sd, 1.0]]).repeat(a.n_curve, 1)
            with torch.no_grad():
                z = sample(g, n=a.n_curve, shape=(a.latent_dim,), y=yv,
                           n_steps=a.ode_steps, method="heun", device=dev)
                ach.append(float(oracle(z).squeeze(-1).mean()) * sd + mu)
        A = np.array(ach); T_ = np.array(targets)
        slope = float(np.polyfit(T_, A, 1)[0])
        r = float(np.corrcoef(T_, A)[0, 1]) if A.std() > 1e-9 else 0.0
        out["curve"][str(w)] = {"targets": targets, "achieved": ach, "slope": slope, "r": r}
        print(f"{w:5.1f} " + "".join(f"{v:9.1f}" for v in A) + f"  {slope:8.3f} {r:7.3f}")

    # quality cost: decode a few cells
    print("\nQUALITY COST at target %.0f C" % a.decode_target)
    trainP, trainC, trainA, ELEM, RH, RM = train_reference(ils, sp["train"])
    print(f"{'w':>5} {'NFE':>5} {'plaus':>7} {'uniq':>7} {'nov_cat':>8} {'achieved':>9}")
    for w in [float(x) for x in a.decode_w.split(",")]:
        g = CFGVelocity(fm, w=w)
        yv = torch.tensor([[(a.decode_target - mu) / sd, 1.0]]).repeat(a.n_decode, 1)
        with torch.no_grad():
            z = sample(g, n=a.n_decode, shape=(a.latent_dim,), y=yv,
                       n_steps=a.ode_steps, method="heun", device=dev)
            got = float(oracle(z).squeeze(-1).mean()) * sd + mu
            smis = decode(ae, sc.inverse(z), tok, 80, repair=False)
        m, pairs = plausibility(smis, len(smis), ELEM, RH, RM)
        m |= novelty(pairs, trainP, trainC, trainA)
        nfe = guided_nfe(2 * a.ode_steps, w)
        out["decoded"][str(w)] = {"nfe": nfe, "achieved_C": got, **{k: m[k] for k in
                                  ("plausible", "uniqueness", "novel_cation_rate")}}
        print(f"{w:5.1f} {nfe:5d} {m['plausible']:7.3f} {m['uniqueness']:7.3f} "
              f"{m['novel_cation_rate']:8.3f} {got:9.1f}")

    write_json(a.out, out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
