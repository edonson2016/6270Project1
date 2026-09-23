"""Experiment 1: how does required latent dimension scale with molecule size?

Splits MOSES into equal-sized bins by SELFIES token length and sweeps the latent
dimension within each bin. Bins are held to the SAME n so the comparison isolates
molecular complexity rather than confounding it with dataset size.

Every config records four arms so the failure modes stay separable:
  fm       z ~ N(0,I) -> ODE -> decode      the latent flow matching model
  prior    z ~ N(0,I) -> decode             the plain VAE, no velocity field
  ceiling  z = encode(real) -> decode       decoder health, upper bound
  plus latent-space diagnostics (active units, aggregate posterior vs prior)

The fm-vs-prior gap is exactly what the velocity field buys.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fm import FlowMatching, Standardizer, TrainConfig, Trainer, VectorDataset, VelocityMLP, sample
from fm.flops import fmt, forward_flops, ode_sampling_flops, seq_forward_flops, train_flops
from fm.latent import SeqVAE

RESULTS = []


def sliced_w2(a, b, n_proj=128, seed=0):
    r = np.random.default_rng(seed)
    d = np.asarray(a).shape[1]
    dirs = r.standard_normal((n_proj, d)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    qs = np.linspace(0, 1, 512)
    pa = np.stack([np.quantile((a @ dirs.T)[:, j], qs) for j in range(n_proj)], 1)
    pb = np.stack([np.quantile((b @ dirs.T)[:, j], qs) for j in range(n_proj)], 1)
    return float(np.sqrt(((pa - pb) ** 2).mean()))


def decode_to_smiles(vae, z, vocab, itos, device):
    import selfies as sf
    toks = vae.generate(z.to(device), vocab["max_len"], vocab["bos"], vocab["eos"]).cpu().numpy()
    out = []
    for row in toks:
        try:
            out.append(sf.decoder("".join(itos[t] for t in row if t > 2)))
        except Exception:
            out.append("")
    return out


def mol_metrics(smiles, train_set, n_req):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    canon = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None:
            canon.append(Chem.MolToSmiles(m))
    uniq = set(canon)
    return {"validity": len(canon) / max(n_req, 1),
            "uniqueness": len(uniq) / max(len(canon), 1),
            "novelty": len(uniq - train_set) / max(len(uniq), 1),
            "n_unique": len(uniq)}


def train_vae(tokens, tokens_in, V, d, args, device):
    vae = SeqVAE(V, latent_dim=d, emb_dim=args.emb_dim, hidden=args.hidden,
                 beta=args.beta, word_dropout=args.word_dropout).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=1e-3)
    ds = TensorDataset(tokens, tokens_in)
    n_val = max(512, len(ds) // 10)
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))
    loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(loader))
    seen = 0
    for ep in range(args.epochs):
        vae.train()
        for tgt, inp in loader:
            tgt, inp = tgt.to(device), inp.to(device)
            o = vae(tgt, inp, tgt)
            opt.zero_grad(set_to_none=True); o["loss"].backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step(); sched.step(); seen += len(tgt)
    vae.eval()
    exact, tot, acc_n, acc_d = 0, 0, 0, 0
    with torch.no_grad():
        for tgt, inp in DataLoader(va, batch_size=512):
            tgt, inp = tgt.to(device), inp.to(device)
            mu, _ = vae.encode(tgt)
            pred = vae.decode(mu, inp).argmax(-1)
            mask = tgt != 0
            exact += int(((pred == tgt) | ~mask).all(1).sum()); tot += len(tgt)
            acc_n += int(((pred == tgt) & mask).sum()); acc_d += int(mask.sum())
    fwd = seq_forward_flops(vae, batch=1, seq_len=tokens.shape[1])
    return vae, exact / tot, acc_n / max(acc_d, 1), train_flops(fwd, seen), fwd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-bin", type=int, default=15_000)
    p.add_argument("--latent-dims", default="16,32,64,128")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--emb-dim", type=int, default=96)
    p.add_argument("--beta", type=float, default=3e-3)
    p.add_argument("--word-dropout", type=float, default=0.25,
                   help="blank this fraction of decoder input tokens; fights posterior collapse")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--fm-steps", type=int, default=6000)
    p.add_argument("--fm-width", type=int, default=384)
    p.add_argument("--fm-depth", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--out", default="runs/exp1")
    args = p.parse_args()

    device = torch.device("cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    vocab = json.loads(Path("data/moses_vocab.json").read_text())
    itos, V, BOS = vocab["itos"], len(vocab["itos"]), vocab["bos"]
    all_tokens = torch.from_numpy(np.load("data/moses_tokens.npy").astype(np.int64))
    lengths = (all_tokens != 0).sum(1).numpy()
    smiles_all = Path("data/moses_train_smiles.txt").read_text().split()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    train_set = set()
    for s in smiles_all:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            train_set.add(Chem.MolToSmiles(m))
    print(f"canonical training set: {len(train_set):,} molecules", flush=True)

    bins = [("short", 0, 32), ("medium", 32, 41), ("long", 41, 999)]
    dims = [int(x) for x in args.latent_dims.split(",")]

    for bname, lo, hi in bins:
        idx = np.where((lengths >= lo) & (lengths < hi))[0]
        rng = np.random.default_rng(0)
        if len(idx) > args.n_per_bin:
            idx = rng.choice(idx, args.n_per_bin, replace=False)
        tok = all_tokens[idx]
        tok_in = torch.cat([torch.full((len(tok), 1), BOS, dtype=torch.long), tok[:, :-1]], 1)
        print(f"\n{'='*72}\nbin '{bname}' [{lo},{hi}): n={len(tok):,}  "
              f"mean len {lengths[idx].mean():.1f}\n{'='*72}", flush=True)

        for d in dims:
            t0 = time.time()
            vae, exact, tok_acc, fl_vae, vae_fwd = train_vae(tok, tok_in, V, d, args, device)
            with torch.no_grad():
                lat = torch.cat([vae.encode(tok[i:i+512].to(device))[0].cpu()
                                 for i in range(0, len(tok), 512)])
            latn = lat.numpy()
            active = int((latn.std(0) > 0.1).sum())
            gprior = np.random.default_rng(0).standard_normal(latn.shape)
            w2_prior = sliced_w2(latn, gprior)

            sc = Standardizer().fit(lat); lz = sc.transform(lat)
            model = FlowMatching(VelocityMLP(d, width=args.fm_width, depth=args.fm_depth))
            cfg = TrainConfig(steps=args.fm_steps, lr=1e-3, out_dir="runs/_tmp_e1",
                              device="cpu", log_every=args.fm_steps, val_every=10**9, ckpt_every=0)
            tr = Trainer(model, DataLoader(VectorDataset(lz), batch_size=256, shuffle=True,
                                           drop_last=True), cfg)
            tr.fit()
            ema = FlowMatching(VelocityMLP(d, width=args.fm_width, depth=args.fm_depth))
            ema.load_state_dict(tr.ema.state_dict()); ema.to(device).eval()
            fl_fm = train_flops(forward_flops(ema.net, 1), args.fm_steps * 256)

            n = args.n_samples
            gz = sample(ema, n=n, shape=(d,), n_steps=args.ode_steps, method="heun", device=device)
            gz_un = sc.inverse(gz.cpu())
            w2_fm = sliced_w2(gz_un.numpy(), latn[:n])

            m_fm = mol_metrics(decode_to_smiles(vae, gz_un, vocab, itos, device), train_set, n)
            zp = torch.randn(n, d)
            m_prior = mol_metrics(decode_to_smiles(vae, zp, vocab, itos, device), train_set, n)
            zc = lat[torch.randperm(len(lat))[:n]]
            m_ceil = mol_metrics(decode_to_smiles(vae, zc, vocab, itos, device), train_set, n)

            dec_fwd = seq_forward_flops(vae.dec, 1, vocab["max_len"]) + \
                      seq_forward_flops(vae.out, 1, vocab["max_len"])
            rec = dict(bin=bname, bin_lo=lo, bin_hi=hi, n=len(tok),
                       mean_len=float(lengths[idx].mean()), latent_dim=d,
                       exact_recon=exact, token_acc=tok_acc, active_units=active,
                       beta=args.beta, word_dropout=args.word_dropout,
                       w2_qz_vs_prior=w2_prior, w2_fm_vs_qz=w2_fm,
                       fm_validity=m_fm["validity"], fm_uniqueness=m_fm["uniqueness"],
                       fm_novelty=m_fm["novelty"], fm_n_unique=m_fm["n_unique"],
                       prior_uniqueness=m_prior["uniqueness"], prior_novelty=m_prior["novelty"],
                       ceil_uniqueness=m_ceil["uniqueness"],
                       flops_vae_train=int(fl_vae), flops_fm_train=int(fl_fm),
                       flops_train_total=int(fl_vae + fl_fm),
                       flops_sample=int(ode_sampling_flops(ema.net, args.ode_steps, "heun") + dec_fwd),
                       secs=round(time.time() - t0, 1))
            RESULTS.append(rec)
            (out / "results.json").write_text(json.dumps(RESULTS, indent=2))
            print(f"  [saved] {bname:<7s} d={d:>4d} exact_recon={exact:.3f} active={active:>3d}/{d} "
                  f"| uniq fm={m_fm['uniqueness']:.3f} prior={m_prior['uniqueness']:.3f} "
                  f"ceil={m_ceil['uniqueness']:.3f} | w2(fm,qz)={w2_fm:.3f} "
                  f"| {rec['secs']:.0f}s", flush=True)

    print(f"\nDONE -> {out/'results.json'}  ({len(RESULTS)} configs)", flush=True)


if __name__ == "__main__":
    main()
