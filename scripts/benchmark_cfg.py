"""Benchmark the conditional model against the unconditional baseline.

To compare fairly with an unconditional flow, the conditional one is asked for
the NATURAL distribution of melting points -- targets drawn from the empirical
train-labelled label distribution rather than one fixed value. That isolates the
cost of conditioning itself from the cost of steering to an unusual target.

Metrics are the ones already in results.jsonl so the rows sit in one table.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm import CFGVelocity, Standardizer, guided_nfe, sample
from fm.il_eval import make_splits, plausibility, novelty, sliced_w2
from fm.il_io import (RESULTS, decode, encode_all, load_ae, load_cfg_fm, load_cond, load_emb,
                      load_ils, load_tokenizer, train_reference, write_json)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="cfg_mp_d16")
    p.add_argument("--ae", default="runs/il/ae/ae_enum_d16.pt")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--weights", default="0,1,2")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--out", default="runs/il/results/cfg_benchmark.json")
    a = p.parse_args()

    dev = torch.device("cpu"); torch.manual_seed(0)
    tok = load_tokenizer()
    ae = load_ae(a.ae, a.latent_dim, tok, 80, dev)

    ils = load_ils()
    sp = make_splits(ils, seed=0)
    mu, sd, raw = load_cond(a.tag, ils)
    lab = raw[sp["train"]][np.isfinite(raw[sp["train"]])]

    Zall = encode_all(ae, load_emb())
    sc = Standardizer().fit(Zall[torch.from_numpy(sp["train"]).long()])
    Z = {k: sc.transform(Zall[torch.from_numpy(v).long()]).numpy() for k, v in sp.items()}

    trainP, trainC, trainA, ELEM, RH, RM = train_reference(ils, sp["train"])

    fm = load_cfg_fm(a.tag, a.latent_dim, dev)

    rng = np.random.default_rng(0)
    out = {}
    print(f"{'arm':>12} {'NFE':>5} {'plaus':>7} {'uniq':>7} {'novcat':>7} "
          f"{'sw2_test':>9} {'sw2_ion':>8} {'sec':>6}")
    print("-" * 70)
    for w in [float(x) for x in a.weights.split(",")]:
        g = CFGVelocity(fm, w=w)
        # targets drawn from the empirical train-labelled distribution
        T = rng.choice(lab, size=a.n, replace=True)
        yv = torch.tensor(np.stack([(T - mu) / sd, np.ones(a.n)], 1)).float()
        t0 = time.time()
        with torch.no_grad():
            z = sample(g, n=a.n, shape=(a.latent_dim,), y=yv,
                       n_steps=a.ode_steps, method="heun", device=dev)
            smis = decode(ae, sc.inverse(z), tok, 80, repair=False)
        secs = time.time() - t0
        m, pairs = plausibility(smis, len(smis), ELEM, RH, RM)
        m |= novelty(pairs, trainP, trainC, trainA)
        zn = z.numpy()
        s_test = sliced_w2(zn, Z["test"]); s_ion = sliced_w2(zn, Z["test_ion"])
        nfe = guided_nfe(2 * a.ode_steps, w)
        out[str(w)] = {"nfe": nfe, "plausible": m["plausible"], "uniqueness": m["uniqueness"],
                       "novel_cation_rate": m["novel_cation_rate"], "sw2_test": s_test,
                       "sw2_test_ion": s_ion, "secs": secs}
        print(f"{'cfg w=' + f'{w:g}':>12} {nfe:5d} {m['plausible']:7.3f} {m['uniqueness']:7.3f} "
              f"{m['novel_cation_rate']:7.3f} {s_test:9.3f} {s_ion:8.3f} {secs:6.0f}")
        (RESULTS / f"samples_cfg_w{w:g}.txt").write_text(
            "\n".join(s for s in smis if s))

    write_json(a.out, out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
