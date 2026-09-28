"""How few integration steps does each model tolerate?

This is the measurement that decides whether OT coupling did anything. Plausibility
at 50 steps can barely move even when the trajectories have improved a great deal,
because 50 Heun steps integrate a curved path accurately too. The payoff of
straighter trajectories shows up at the LOW end: a straight path is exactly
integrable in one Euler step, a curved one is not.

So this sweeps Euler steps -- first order, therefore maximally sensitive to
curvature -- and scores each cloud with sliced-W2 against the real test latents.
No decoding, so the whole grid costs seconds.

`euler_1` is the straightness number: x1_hat = x0 + v(x0, 0). The closer that is
to the 50-step result, the straighter the learned field.

Sources are re-fitted rather than loaded; EM is deterministic given its seed and
the train cloud, so this reproduces what training used.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm.nets import make_velocity_net
from fm import FlowMatching, Standardizer, sample
from fm.il_eval import make_splits, sliced_w2
from fm.il_io import (encode_all, load_ae, load_emb, load_ils, load_results, load_tokenizer,
                      write_json)
from fm.source import make_source


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tags", nargs="+")
    p.add_argument("--ae", default="runs/il/ae/ae_enum_d16.pt")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--steps", default="1,2,4,8,16,50")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--method", default="euler", choices=["euler", "heun"])
    p.add_argument("--tokenizer", default="bpe", choices=["bpe", "selfies"])
    p.add_argument("--max-len", type=int, default=80)
    p.add_argument("--selfies-vocab", default="data/selfies_vocab.json")
    p.add_argument("--ref-split", default="test", choices=["valid", "test", "test_ion"],
                   help="which real latents to score against. 'valid' is the honest choice when "
                        "this sweep is being used to SELECT a step count; 'test' reports it.")
    p.add_argument("--out", default="runs/il/results/nfe_sweep.json")
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

    rows = load_results()
    steps = [int(x) for x in a.steps.split(",")]
    out = {}
    hdr = f"{'tag':>11} {'source':>7} {'coupling':>11} | " + "".join(f"{f'{s}st':>8}" for s in steps) + f" | {'1/50':>6}"
    print(f"sliced-W2 to real {a.ref_split} latents, {a.method} integration, n={a.n}\n")
    print(hdr); print("-" * len(hdr))

    for tag in a.tags:
        r = rows[tag]
        kind = r.get("source", "gauss"); cpl = r.get("coupling", "independent")
        torch.manual_seed(0)
        src = None if kind == "gauss" else make_source(kind, k=r.get("gmm_k", 8),
                                                       nu=r.get("nu", 5.0)).fit(sc.transform(Ztr))
        net = make_velocity_net(r.get("fm_arch", "mlp"), a.latent_dim, width=384)
        fm = FlowMatching(net, source=src)
        fm.load_state_dict(torch.load(f"runs/il/_fm_{tag}/last.pt", map_location=dev,
                                      weights_only=False)["ema"], strict=False)
        fm.eval()
        torch.manual_seed(0)
        x0 = (src.sample(a.n, device=dev) if src is not None
              else torch.randn(a.n, a.latent_dim))
        vals = []
        for st in steps:
            with torch.no_grad():
                z = sample(fm, x0=x0.clone(), n_steps=st, method=a.method, device=dev)
            vals.append(sliced_w2(z.numpy(), ref))
        ratio = vals[0] / vals[-1] if vals[-1] > 0 else float("nan")
        out[tag] = {"source": kind, "coupling": cpl, "steps": steps, "sw2": vals,
                    "straightness_1_over_50": ratio}
        print(f"{tag:>11} {kind:>7} {cpl:>11} | " + "".join(f"{v:8.3f}" for v in vals)
              + f" | {ratio:6.2f}")

    write_json(a.out, out)
    print(f"\n-> {a.out}   (1/50 near 1.0 = straight trajectories)")


if __name__ == "__main__":
    main()
