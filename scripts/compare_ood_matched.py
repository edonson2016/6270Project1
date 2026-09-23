"""Compare sw2_ood across runs against a MATCHED reference set.

This exists because the raw §6.10 comparison was a trap. Moving 150 OOD pairs
into train shrank the OOD eval set from 227 to 64, and sw2 on 64 points carries
about +0.05 of finite-sample inflation -- enough to turn a small improvement
(0.538 -> 0.513) into an apparent regression (0.536 -> 0.647). Any run whose
`ood` split differs in SIZE from the one it is being compared to has to come
through here, not out of results.jsonl.

Each run is re-encoded with its own autoencoder and scored against the same two
reference clouds: all 227 original OOD pairs, and the held-out 64. `sliced_w2`
standardizes to the reference cloud's units, which is what makes clouds from
different latent spaces comparable at all. The random-subset column is the
control: 64-point subsamples of the SAME 227 cloud, so it prices the inflation
directly rather than assuming it.

  scripts/compare_ood_matched.py enum_fm_d16:runs/il/ae/ae_enum_d16.pt ...
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm import FlowMatching, DDPM, Standardizer, VelocityMLP, sample
from fm.il_pipeline import LatentAE
from fm.il_eval import make_splits, make_splits_extended, sliced_w2

DATA = Path("data"); MODEL = "seyonec/ChemBERTa-zinc-base-v1"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("arms", nargs="+", help="tag:ae_checkpoint")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--n-subsets", type=int, default=20)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--out", default="runs/il/results/ood_matched.json")
    a = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    dev = torch.device("cpu")

    rows = [json.loads(l) for l in open("runs/il/results/results.jsonl")]
    by_tag = {}
    for r in rows:                       # a __sampler suffix shares one trained model
        by_tag.setdefault(r["tag"].split("__")[0], r)

    ood_all = json.loads((DATA/"il_ood_pairs.json").read_text())
    E_ood = torch.from_numpy(np.load(DATA/"il_emb_ood.npy")).float()
    held = {q["pair"] for q in json.loads((DATA/"il_ood_pairs_v4.json").read_text())}
    ho_idx = [i for i, q in enumerate(ood_all) if q["pair"] in held]
    print(f"reference clouds: all {len(ood_all)}, held-out {len(ho_idx)}\n", flush=True)

    out = {}
    for arm in a.arms:
        tag, aep = arm.split(":", 1)
        r = by_tag[tag]; D = r["latent_dim"]
        torch.manual_seed(0)

        ck = torch.load(aep, map_location=dev, weights_only=False)
        m = LatentAE(768, D, tok.vocab_size, 80, decoder=r["decoder"], hidden=512,
                     pad_id=tok.pad_token_id)
        m.load_state_dict(ck["model"]); m.eval()

        stem = r.get("corpus") or "il_pairs_v2"
        ils = json.loads((DATA/f"{stem}.json").read_text())
        esuf = "" if stem == "il_pairs_v2" else "_" + stem.rsplit("_", 1)[-1]
        E = torch.from_numpy(np.load(DATA/f"il_emb_ils{esuf}.npy")).float()
        n0 = r.get("n_original") or (4790 if esuf else 0)
        sp = (make_splits_extended(ils, n0) if n0 else make_splits(ils))

        with torch.no_grad():
            Z = torch.cat([m.encode(E[i:i+512]) for i in range(0, len(E), 512)])
            Zo = torch.cat([m.encode(E_ood[i:i+512]) for i in range(0, len(E_ood), 512)])

        # the flow was standardized against its own training cloud, upweighting included
        Ztr = Z[torch.from_numpy(sp["train"]).long()]
        if r.get("upweight", 1) > 1:
            src = np.array([ils[i].get("source", "") == r["upweight_source"]
                            for i in sp["train"]])
            Ztr = torch.cat([Ztr, Ztr[torch.from_numpy(
                np.repeat(np.where(src)[0], r["upweight"] - 1)).long()]])
        sc = Standardizer().fit(Ztr)

        net = VelocityMLP(D, width=384, depth=4)
        gen = (FlowMatching(net) if r["generative"] == "fm"
               else DDPM(net, T=r["ddpm_T"], schedule=r.get("ddpm_schedule", "linear")))
        gen.load_state_dict(torch.load(f"runs/il/_fm_{tag}/last.pt", map_location=dev,
                                       weights_only=False)["ema"])
        gen.eval()
        if r["generative"] == "fm":
            raw = sample(gen, n=a.n_samples, shape=(D,), n_steps=a.ode_steps,
                         method=r["sampler"], device=dev)
        else:
            raw = gen.sample(a.n_samples, (D,), device=dev, sampler=r["sampler"],
                             n_steps=a.ode_steps, clip_x0=r.get("clip_x0"))
        gz = sc.inverse(raw.cpu()).numpy(); Zo = Zo.numpy()

        all_n = sliced_w2(gz, Zo)
        only  = sliced_w2(gz, Zo[ho_idx])
        rng = np.random.default_rng(0)
        sub = [sliced_w2(gz, Zo[rng.choice(len(Zo), len(ho_idx), replace=False)])
               for _ in range(a.n_subsets)]
        out[tag] = {"sw2_all": all_n, "sw2_heldout": only,
                    "subset_mean": float(np.mean(sub)), "subset_std": float(np.std(sub)),
                    "inflation": float(np.mean(sub)) - all_n, "n_ref_all": len(Zo),
                    "n_ref_heldout": len(ho_idx)}
        print(f"{tag:22} sw2 vs all {len(Zo)} = {all_n:.3f}   vs held-out "
              f"{len(ho_idx)} = {only:.3f}", flush=True)
        print(f"{'':22} random {len(ho_idx)}-subsets: {np.mean(sub):.3f} +/- {np.std(sub):.3f}"
              f"  (finite-sample inflation {np.mean(sub)-all_n:+.3f})\n", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
