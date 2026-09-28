"""Sample a conditional model at a fixed property target and write the SMILES."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm import CFGVelocity, DDPM, FlowMatching, Standardizer, VelocityMLP, sample
from fm.il_eval import make_splits
from fm.il_io import decode, encode_all, load_ae, load_cond, load_emb, load_ils, load_tokenizer

p = argparse.ArgumentParser()
p.add_argument("--tag", default="sf_cfg_mp")
p.add_argument("--ae", default="runs/il/ae/ae_sf_d16.pt")
p.add_argument("--target", type=float, default=250.0)
p.add_argument("--w", type=float, default=1.0)
p.add_argument("--n", type=int, default=2000)
p.add_argument("--ode-steps", type=int, default=50)
p.add_argument("--generative", default="fm", choices=["fm", "ddpm"])
p.add_argument("--sampler", default="", help="ddpm: ancestral|ddim")
p.add_argument("--ddpm-T", type=int, default=1000)
p.add_argument("--clip-x0", type=float, default=3.0)
p.add_argument("--out", default="")
a = p.parse_args()

dev = torch.device("cpu"); torch.manual_seed(0)
tok = load_tokenizer("selfies")
ae = load_ae(a.ae, 16, tok, 96, dev)
ils = load_ils(); sp = make_splits(ils, seed=0)
Z = encode_all(ae, load_emb())
sc = Standardizer().fit(Z[torch.from_numpy(sp["train"]).long()])
mu, sd, _ = load_cond(a.tag, ils)
net = VelocityMLP(16, cond_dim=2, width=384, depth=4)
gen = (FlowMatching(net) if a.generative == "fm" else DDPM(net, T=a.ddpm_T, schedule="linear"))
gen.load_state_dict(torch.load(f"runs/il/_fm_{a.tag}/last.pt", map_location=dev,
                               weights_only=False)["ema"]); gen.eval()
y = torch.tensor([[(a.target - mu) / sd, 1.0]]).repeat(a.n, 1)
with torch.no_grad():
    if a.generative == "fm":
        z = sample(CFGVelocity(gen, w=a.w), n=a.n, shape=(16,), y=y,
                   n_steps=a.ode_steps, method="heun", device=dev)
    else:
        z = gen.sample(a.n, (16,), y=y, device=dev, sampler=(a.sampler or "ancestral"),
                       n_steps=a.ode_steps, clip_x0=a.clip_x0, guide_w=a.w)
out = decode(ae, sc.inverse(z), tok, 96)
f = a.out or f"runs/il/results/samples_{a.tag}_T{int(a.target)}_w{a.w:g}.txt"
Path(f).write_text("\n".join(s for s in out if s))
print(f"wrote {sum(1 for s in out if s)} SMILES -> {f}  (target {a.target} C, w={a.w})")
