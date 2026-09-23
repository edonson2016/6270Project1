"""Experiment 2: latent FM vs hand-chosen features on molecular conformers.

The question: when you replace a hand-designed featurization with a learned
embedding, what do you pay in FLOPs and what do you get in accuracy?

Conformers are the right testbed because the answer is known in advance. A
molecule with N atoms has exactly 3N-6 internal degrees of freedom, so the data
lies on a manifold of that dimension. A latent smaller than 3N-6 MUST lose
physics; a latent larger than it cannot gain any. If the sweep shows a knee at
3N-6, the method is measuring what we think it is measuring.

Arms, all evaluated with the same geometry metrics:
  - hand-featurized, no autoencoder:  aligned Cartesian (3N), pairwise distance
  - learned latent:                   AE 3N -> d -> 3N, FM in the d-dim latent

    .venv/bin/python scripts/exp2_conformer_pareto.py --molecules ethanol,toluene,aspirin
"""
from __future__ import annotations

import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fm import FlowMatching, Standardizer, TrainConfig, Trainer, VectorDataset, VelocityMLP, sample
from fm.autoencoder import MLPAutoencoder
from fm.featurize import AlignedCartesian, PairwiseDistance, geometry_report, infer_bonds
from fm.flops import fmt, forward_flops, ode_sampling_flops, train_flops

RESULTS = []


def log_result(out_dir: Path, rec: dict) -> None:
    """Append and flush after every config, so a crash never loses finished work."""
    RESULTS.append(rec)
    (out_dir / "results.json").write_text(json.dumps(RESULTS, indent=2))
    dtag = rec.get("latent_dim")
    dtag = "-" if dtag is None else str(dtag)
    print(f"  [saved] {rec['arm']:<22s} d={dtag:>4s} "
          f"valid={rec['frac_valid']:.4f} std_ratio={rec['bond_std_ratio']:.4f} "
          f"W={rec.get('width',0):>4d} sampleFLOPs={fmt(rec['flops_sample'])}", flush=True)


def train_ae(x: torch.Tensor, d: int, epochs: int, width: int, device, bs: int = 512):
    D = x.shape[1]
    ae = MLPAutoencoder(D, d, width=width, depth=3, beta=0.0).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=2e-3)
    loader = DataLoader(TensorDataset(x), batch_size=bs, shuffle=True, drop_last=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(loader))
    seen = 0
    for ep in range(epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            o = ae(xb)
            opt.zero_grad(set_to_none=True); o["loss"].backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step(); sched.step(); seen += len(xb)
    ae.eval()
    with torch.no_grad():
        recon = float((ae(x[:20000].to(device))["recon"]))
    ae_fwd = forward_flops(ae, batch=1)
    return ae, recon, train_flops(ae_fwd, seen), ae_fwd


def run_fm(x1: torch.Tensor, steps: int, width: int, depth: int, device, bs: int = 256):
    D = x1.shape[1]
    net = VelocityMLP(D, width=width, depth=depth)
    model = FlowMatching(net)
    loader = DataLoader(VectorDataset(x1), batch_size=bs, shuffle=True, drop_last=True)
    cfg = TrainConfig(steps=steps, lr=1e-3, out_dir=str(Path("runs/_tmp_fm")),
                      device=str(device), log_every=max(steps // 2, 1),
                      val_every=10**9, ckpt_every=0)
    tr = Trainer(model, loader, cfg)
    tr.fit()
    ema = FlowMatching(VelocityMLP(D, width=width, depth=depth))
    ema.load_state_dict(tr.ema.state_dict()); ema.to(device).eval()
    fm_fwd = forward_flops(net, batch=1)
    return ema, train_flops(fm_fwd, steps * bs), net


def evaluate(coords_gen, ref, z, bonds) -> dict:
    return geometry_report(coords_gen, ref, z, bonds)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--molecules", default="ethanol")
    p.add_argument("--n", type=int, default=90_000)
    p.add_argument("--fm-steps", type=int, default=5000)
    p.add_argument("--ae-epochs", type=int, default=12)
    p.add_argument("--widths", default="128,256",
                   help="velocity-field widths to sweep; this is the FLOPs axis")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--ae-width", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--out", default="runs/exp2")
    args = p.parse_args()

    device = torch.device("cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    for mol in args.molecules.split(","):
        raw = np.load(f"data/rmd17_{mol}.npz")
        coords = raw["coords"][: args.n]
        zc = raw["nuclear_charges"]
        N = len(zc); dof = 3 * N - 6
        bonds = infer_bonds(coords, zc)
        ref = coords[:5000] - coords[:5000].mean(axis=1, keepdims=True)
        print(f"\n{'='*70}\n{mol}: N={N} atoms, 3N={3*N}, true DOF 3N-6={dof}, "
              f"{len(bonds)} bonds, {len(coords):,} conformers\n{'='*70}", flush=True)

        # ---------------- hand-featurized arms (no autoencoder) ----------------
        widths = [int(w) for w in args.widths.split(",")]
        for fname, F in (("aligned", AlignedCartesian()), ("distance", PairwiseDistance())):
            F.fit(coords); xf = F.transform(coords)
            xt = torch.as_tensor(xf)
            sc = Standardizer().fit(xt); xs = sc.transform(xt)
          
            for W in widths:
                t0 = time.time()
                ema, fl_train, net = run_fm(xs, args.fm_steps, W, args.depth, device)
                g = sample(ema, n=args.n_samples, shape=(xs.shape[1],),
                           n_steps=args.ode_steps, method="heun", device=device)
                g = sc.inverse(g.cpu()).numpy()
                gen = F.inverse(g)
                rep = evaluate(gen, ref, zc, bonds)
                rep.update(molecule=mol, n_atoms=N, dof=dof, arm=f"handcrafted:{fname}",
                           latent_dim=None, feat_dim=int(xs.shape[1]), ae_recon=0.0, width=W,
                           flops_ae_train=0, flops_fm_train=int(fl_train),
                           flops_train_total=int(fl_train),
                           flops_sample=int(ode_sampling_flops(net, args.ode_steps, "heun")),
                           secs=round(time.time() - t0, 1))
                log_result(out, rep)

        # ---------------- learned-latent arms ----------------
        F = AlignedCartesian(); F.fit(coords)
        xf = torch.as_tensor(F.transform(coords))
        sc = Standardizer().fit(xf); xs = sc.transform(xf)
        D = xs.shape[1]
        dims = sorted({max(2, int(round(dof * f))) for f in (0.25, 0.5, 0.75, 1.0, 1.25)})
        print(f"  latent dims to sweep: {dims}   (true DOF = {dof}, ambient = {D})", flush=True)

        for d in dims:
            # The AE is trained once per latent dim and reused across velocity widths.
            ae, ae_recon, fl_ae, ae_fwd = train_ae(xs, d, args.ae_epochs, args.ae_width, device)
            print(f"    AE d={d:>3d}  recon MSE {ae_recon:.6f}", flush=True)
            with torch.no_grad():
                lat = torch.cat([ae.encode(xs[i:i+4096].to(device))[0].cpu()
                                 for i in range(0, len(xs), 4096)])
            lsc = Standardizer().fit(lat); lz = lsc.transform(lat)
            dec_fwd = forward_flops(ae.decoder, batch=1)

            for W in widths:
                t0 = time.time()
                ema, fl_fm, net = run_fm(lz, args.fm_steps, W, args.depth, device)
                gz = sample(ema, n=args.n_samples, shape=(d,), n_steps=args.ode_steps,
                            method="heun", device=device)
                with torch.no_grad():
                    xh = ae.decode(lsc.inverse(gz.cpu()).to(device)).cpu()
                gen = F.inverse(sc.inverse(xh).numpy())
                rep = evaluate(gen, ref, zc, bonds)
                rep.update(molecule=mol, n_atoms=N, dof=dof, arm="latent", latent_dim=d,
                           feat_dim=D, ae_recon=ae_recon, width=W,
                           flops_ae_train=int(fl_ae), flops_fm_train=int(fl_fm),
                           flops_train_total=int(fl_ae + fl_fm),
                           flops_sample=int(ode_sampling_flops(net, args.ode_steps, "heun") + dec_fwd),
                           secs=round(time.time() - t0, 1))
                log_result(out, rep)

    print(f"\nDONE -> {out/'results.json'}  ({len(RESULTS)} configs)", flush=True)


if __name__ == "__main__":
    main()
