"""Does validation-based checkpoint selection change anything in Stage 4?

Every run in this study samples from the FINAL EMA weights. A `best.pt` is written
whenever validation FM loss improves, but it is never loaded -- so there is no
early stopping anywhere, and the rubric question "was the model selected on
validation?" has to be answered "no". This script measures what that costs.

For each tag it scores BOTH checkpoints under identical sampling:

  last.pt   final EMA  -- what every reported number used
  best.pt   EMA at the step with the lowest validation FM loss

and reports plausibility / uniqueness / novelty plus sliced-W2 against valid,
test and test_ion. If the two agree, "no early stopping" is a non-issue and can
be stated as such; if best.pt wins, every headline number is leaving something
on the table.

Also reports the labelled-only sw2 reference for conditional runs, which is the
control the CFG benchmark was missing: conditional arms reproduce the labelled
sub-population, while sw2 is normally measured against the full split.

    .venv/bin/python scripts/ckpt_select.py sf_sig010 sf_base \
        --ae runs/il/ae/ae_sf_d16.pt --tokenizer selfies --max-len 96
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm import FlowMatching, DDPM, Standardizer, CondOTPath, sample
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
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--out", default="runs/il/results/ckpt_select.json")
    a = p.parse_args()

    dev = torch.device("cpu")
    tok = load_tokenizer(a.tokenizer, a.selfies_vocab)
    ae = load_ae(a.ae, a.latent_dim, tok, a.max_len, dev)

    ils = load_ils()
    sp = make_splits(ils, seed=0)
    Zall = encode_all(ae, load_emb())
    Ztr = Zall[torch.from_numpy(sp["train"]).long()]
    sc = Standardizer().fit(Ztr)
    REF = {k: sc.transform(Zall[torch.from_numpy(sp[k]).long()]).numpy()
           for k in ("valid", "test", "test_ion")}

    trainP, trainC, trainA, ELEM, RH, RM = train_reference(ils, sp["train"])

    rows = load_results()
    out = {}
    print(f"{'tag':>14} {'ckpt':>6} {'plaus':>7} {'uniq':>7} {'n_uniq':>7} "
          f"{'nov_cat':>8} {'sw2_val':>8} {'sw2_test':>9} {'sw2_ion':>8}")
    print("-" * 82)

    for tag in a.tags:
        r = rows.get(tag, {})
        kind = r.get("source", "gauss")
        arch = r.get("fm_arch", "mlp")
        gen_kind = r.get("generative", "fm")
        src = (None if kind == "gauss" else
               make_source(kind, k=r.get("gmm_k", 8), nu=r.get("nu", 5.0)).fit(sc.transform(Ztr)))
        res = {}
        for name in ("last.pt", "best.pt"):
            f = Path(f"runs/il/_fm_{tag}/{name}")
            if not f.exists():
                print(f"{tag:>14} {name:>6}   (missing)")
                continue
            net = make_velocity_net(arch, a.latent_dim, width=384)
            model = (FlowMatching(net, path=CondOTPath(sigma_min=r.get("sigma_min") or 0.0),
                                 source=src, coupling=r.get("coupling", "independent"))
                     if gen_kind == "fm" else DDPM(net, T=r.get("ddpm_T", 1000)))
            model.load_state_dict(torch.load(f, map_location=dev, weights_only=False)["ema"],
                                  strict=False)
            model.eval()
            torch.manual_seed(0)
            if gen_kind == "fm":
                x0 = (src.sample(a.n, device=dev) if src is not None
                      else torch.randn(a.n, a.latent_dim))
                with torch.no_grad():
                    raw = sample(model, x0=x0, n_steps=a.ode_steps, method="heun", device=dev)
            else:
                with torch.no_grad():
                    raw = model.sample(a.n, (a.latent_dim,), device=dev,
                                       sampler="ancestral", clip_x0=3.0)
            # REF is standardized, so sw2 must be taken in the standardized frame;
            # decoding needs the raw frame. Mixing them silently returns garbage.
            sw = {k: sliced_w2(raw.cpu().numpy(), REF[k]) for k in REF}
            gz = sc.inverse(raw.cpu())
            m, pairs = plausibility(decode(ae, gz, tok, a.max_len, dev), a.n, ELEM, RH, RM)
            m |= novelty(pairs, trainP, trainC, trainA)
            res[name] = {k: m[k] for k in ("plausible", "uniqueness", "n_unique",
                                           "novel_cation_rate")} | {"sw2": sw}
            print(f"{tag:>14} {name.split('.')[0]:>6} {m['plausible']:>7.3f} "
                  f"{m['uniqueness']:>7.3f} {m['n_unique']:>7d} {m['novel_cation_rate']:>8.3f} "
                  f"{sw['valid']:>8.3f} {sw['test']:>9.3f} {sw['test_ion']:>8.3f}")
        if "last.pt" in res and "best.pt" in res:
            d = res["best.pt"]["plausible"] - res["last.pt"]["plausible"]
            res["delta_plausible_best_minus_last"] = d
            print(f"{'':>14} {'Δ':>6} {d:>+7.3f}   (best − last)")
        out[tag] = res

    write_json(a.out, out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
