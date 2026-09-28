"""Decoded quality AND diversity across the NFE ladder -- the headline evidence.

scripts/nfe_sweep.py measures straightness in LATENT space (sliced-W2) and stops
there. That is the right diagnostic for the mechanism, but it cannot support the
claim the recipe is actually making, which is about decoded molecules:

    high uniqueness and novelty at LOW sampling cost.

Every run logged so far sampled at heun-50 (100 NFE), so the recipe and the
baseline have only ever been compared at EQUAL, HIGH cost -- where they are
nearly tied on diversity. The interesting regime is 1-16 NFE, where OT's
straightening is supposed to let the recipe hold its diversity while the
baseline falls apart. This script measures that directly: sample, decode, score.

Sampling only -- no training. Reads the EMA weights of runs already logged.

    .venv/bin/python scripts/nfe_decoded.py sf_base sf_sig010 sf_sig020 \
        --ae runs/il/ae/ae_sf_d16.pt --steps 1,2,4,8,16,50
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm import FlowMatching, Standardizer, CondOTPath, sample
from fm.nets import make_velocity_net
from fm.source import make_source
from fm.il_eval import make_splits, plausibility, novelty, sliced_w2
from fm.il_io import (decode, encode_all, load_ae, load_emb, load_ils, load_results,
                      load_tokenizer, train_reference, write_json)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tags", nargs="+")
    p.add_argument("--ae", default="runs/il/ae/ae_sf_d16.pt")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--tokenizer", default="selfies", choices=["bpe", "selfies"])
    p.add_argument("--max-len", type=int, default=96)
    p.add_argument("--selfies-vocab", default="data/selfies_vocab.json")
    p.add_argument("--steps", default="1,2,4,8,16,50")
    p.add_argument("--method", default="euler", choices=["euler", "heun"])
    p.add_argument("--also-heun50", action="store_true",
                   help="add the heun-50 / 100-NFE reference row for each tag")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--ref-split", default="test", choices=["valid", "test", "test_ion"])
    p.add_argument("--out", default="runs/il/results/nfe_decoded.json")
    a = p.parse_args()

    dev = torch.device("cpu")
    tok = load_tokenizer(a.tokenizer, a.selfies_vocab)
    ae = load_ae(a.ae, a.latent_dim, tok, a.max_len, dev)

    ils = load_ils()
    sp = make_splits(ils, seed=0)
    Zall = encode_all(ae, load_emb())
    Ztr = Zall[torch.from_numpy(sp["train"]).long()]
    sc = Standardizer().fit(Ztr)
    ref = sc.transform(Zall[torch.from_numpy(sp[a.ref_split]).long()]).numpy()

    trainP, trainC, trainA, ELEM, RH, RM = train_reference(ils, sp["train"])

    rows = load_results()
    steps = [int(x) for x in a.steps.split(",")]
    plan = [(a.method, s) for s in steps] + ([("heun", 50)] if a.also_heun50 else [])

    out = {}
    hdr = (f"{'tag':>12} {'solver':>9} {'NFE':>5} | {'plaus':>7} {'uniq':>7} {'n_uniq':>7} "
           f"{'nov_cat':>8} | {'sw2':>6} {'sec':>6}")
    print(f"decoded NFE ladder, n={a.n}, sw2 vs real {a.ref_split} latents\n")
    print(hdr); print("-" * len(hdr))

    for tag in a.tags:
        r = rows.get(tag, {})
        kind = r.get("source", "gauss")
        src = (None if kind == "gauss" else
               make_source(kind, k=r.get("gmm_k", 8), nu=r.get("nu", 5.0)).fit(sc.transform(Ztr)))
        net = make_velocity_net(r.get("fm_arch", "mlp"), a.latent_dim, width=384)
        fm = FlowMatching(net, path=CondOTPath(sigma_min=r.get("sigma_min") or 0.0),
                          source=src, coupling=r.get("coupling", "independent"))
        fm.load_state_dict(torch.load(f"runs/il/_fm_{tag}/last.pt", map_location=dev,
                                      weights_only=False)["ema"], strict=False)
        fm.eval()
        out[tag] = {"source": kind, "coupling": r.get("coupling", "independent"),
                    "sigma_min": r.get("sigma_min"), "rows": []}

        for method, st in plan:
            torch.manual_seed(0)
            x0 = (src.sample(a.n, device=dev) if src is not None
                  else torch.randn(a.n, a.latent_dim))
            t0 = time.time()
            with torch.no_grad():
                z = sample(fm, x0=x0, n_steps=st, method=method, device=dev)
            secs = time.time() - t0
            # sw2 is measured in the STANDARDIZED frame, because `ref` is standardized.
            # Decoding needs the raw frame. Mixing the two silently returns garbage.
            sw2 = sliced_w2(z.cpu().numpy(), ref)
            gz = sc.inverse(z.cpu())
            m, pairs = plausibility(decode(ae, gz, tok, a.max_len, dev), a.n, ELEM, RH, RM)
            m |= novelty(pairs, trainP, trainC, trainA)
            nfe = st * (2 if method == "heun" else 1)
            row = {"method": method, "steps": st, "nfe": nfe,
                   "plausible": m["plausible"], "uniqueness": m["uniqueness"],
                   "n_unique": m["n_unique"], "novel_cation_rate": m["novel_cation_rate"],
                   "sw2": sw2, "sample_secs": secs}
            out[tag]["rows"].append(row)
            print(f"{tag:>12} {method:>9} {nfe:>5} | {m['plausible']:>7.3f} "
                  f"{m['uniqueness']:>7.3f} {m['n_unique']:>7d} {m['novel_cation_rate']:>8.3f} "
                  f"| {row['sw2']:>6.3f} {secs:>6.1f}")

    write_json(a.out, out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
