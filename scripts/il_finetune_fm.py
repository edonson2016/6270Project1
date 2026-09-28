"""Stages 2-5: fine-tune on ILs, encode, flow match, generate, score.

  Stage 2  fine-tune MLP_down + decoder on the TRAIN split only
  Stage 3  encode every split once -> fixed z clouds
  Stage 4  train the velocity field on the TRAIN z cloud. The decoder gets NO gradient.
  Stage 5  generate and score, against held-out and out-of-distribution sets

Splits (fm/il_eval.make_splits), all disjoint:

  train     ~72%  fits the autoencoder and the flow
  valid     ~9%   the ONLY set used for the gate and for monitoring
  test      ~9%   held-out pairs, ions mostly seen -- tests recombination
  test_ion  ~10%  held out by whole CATION, so no cation appears in train --
                  the real generalization test
  ood       227   ILThermo/PubChem pairs absent from the Zenodo corpus, from a
                  different curation pipeline. A second test set, touched only
                  at final scoring.

Arms. `fm` and `prior` are unconditional, so they have no split; the ceiling and
real arms are computed PER evaluation set, which is what makes AE generalization
visible separately from flow quality:

  fm            z ~ N(0,I) -> ODE -> decode      the model
  prior         z ~ N(0,I) -> decode             what the velocity field adds
  ceiling_<s>   z = encode(real from s) -> decode    decoder health on s
  real_<s>      the actual SMILES of s             calibration: must score ~1.0

--latent-mode random assigns each IL a fixed random z instead of encoding it.
That is the control for whether latent ORGANIZATION matters.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset

import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm import (CFGVectorDataset, DDPM, FlowMatching, Standardizer, TrainConfig, Trainer,
                VectorDataset, make_velocity_net, n_function_evals, sample)
from fm.flops import forward_flops, sampling_flops, train_flops
from fm.il_pipeline import LatentAE
from fm.source import make_source
from fm.paths import CondOTPath
from fm.il_eval import (make_splits, make_splits_extended, ion_sets, plausibility,
                        element_set, heavy_and_mw, novelty, recovery, sliced_w2)
from fm.il_io import DATA, decode, load_tokenizer


@torch.no_grad()
def recon_acc(model, E, T_in, T, dev, zfix=None, bs=256):
    """Teacher-forced reconstruction on a split -- the cheapest AE generalization read."""
    acc = ex = n = 0
    for i in range(0, len(E), bs):
        zo = None if zfix is None else zfix[i:i+bs].to(dev)
        o = model.loss(E[i:i+bs].to(dev), T_in[i:i+bs].to(dev), T[i:i+bs].to(dev), z_override=zo)
        acc += float(o["token_acc"]); ex += float(o["exact"]); n += 1
    return acc/max(n,1), ex/max(n,1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default="runs/il/pretrain/pretrained.pt")
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--decoder", default="ar", choices=["ar", "nar"])
    p.add_argument("--latent-mode", default="chemberta", choices=["chemberta", "random"])
    p.add_argument("--enum-pairs", default="",
                   help="Phase 2a: enumerated decoder corpus tag (e.g. 'enum'). Empty = skip 2a.")
    p.add_argument("--enum-epochs", type=int, default=3)
    p.add_argument("--enum-lr", type=float, default=1e-3)
    p.add_argument("--save-ae", default="", help="write the fine-tuned autoencoder here")
    p.add_argument("--load-ae", default="",
                   help="load a fine-tuned AE and SKIP phases 2a/2b (reuse across fm/ddpm)")
    p.add_argument("--ft-epochs", type=int, default=40)
    p.add_argument("--ft-lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--max-len", type=int, default=80)
    p.add_argument("--word-dropout", type=float, default=0.25)
    p.add_argument("--generative", default="fm", choices=["fm", "ddpm"],
                   help="Stage 4 generative model. Everything else is identical.")
    p.add_argument("--ddpm-T", type=int, default=1000)
    p.add_argument("--ddpm-schedule", default="linear", choices=["linear", "cosine"])
    p.add_argument("--sampler", default="",
                   help="comma list. fm: euler|heun|rk4 (default heun). "
                        "ddpm: ancestral|ddim (default ancestral). Several samplers reuse "
                        "ONE trained model and emit one results row each, which is what "
                        "isolates sampler cost from training.")
    p.add_argument("--clip-x0", type=float, default=None,
                   help="DDPM only: clamp predicted x0. Needed for the cosine schedule.")
    p.add_argument("--fm-steps", type=int, default=8000)
    p.add_argument("--fm-width", type=int, default=384)
    p.add_argument("--fm-arch", default="mlp", choices=["mlp", "ffnn", "cnn", "unet"],
                   help="velocity/noise network. 'mlp' is VelocityMLP; the others are the "
                        "parameter-matched ablation in fm/nets.py. --fm-width applies to mlp only.")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--gate", type=float, default=0.90)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0,
                   help="global + Stage 4 seed. Splits are governed by --split-seed and do "
                        "not move with this, so replicates share identical data.")
    p.add_argument("--corpus", default="il_pairs_v2",
                   help="observed-pair corpus stem under data/ (v3 = Zenodo + CIR ILThermo)")
    p.add_argument("--ood-stem", default="il_ood_pairs",
                   help="held-out OOD eval set; _vN variants shrink it when part went to train")
    p.add_argument("--upweight", type=float, default=1.0,
                   help="replicate the --upweight-source rows this many times in the "
                        "FLOW's training cloud only. The autoencoder is untouched, so "
                        "with --load-ae the sampling density is the ONLY variable.")
    p.add_argument("--upweight-source", default="",
                   help="corpus `source` field to upweight, e.g. ilthermo_ood_train")
    p.add_argument("--train-subset-n", type=int, default=0,
                   help="train the FLOW on a random subset of this many train latents "
                        "(autoguidance: builds a deliberately weaker guiding model). The "
                        "Standardizer is then fit on the FULL train cloud, so the weak and "
                        "strong fields share one coordinate frame and v1-v0 is meaningful.")
    p.add_argument("--train-subset-seed", type=int, default=0)
    p.add_argument("--cond-prop", default="",
                   help="condition the flow on a numeric property, e.g. 'mp'. Builds a "
                        "2-channel conditioning vector [standardized value, mask]; the mask "
                        "channel is required because a standardized 0 is the MEAN label, not "
                        "the absence of one. Unlabelled pairs carry mask 0 and train only the "
                        "unconditional branch.")
    p.add_argument("--cfg-dropout", type=float, default=0.15,
                   help="probability a LABELLED example is shown with the null token, so one "
                        "network learns both p(x|y) and p(x)")
    p.add_argument("--source", default="gauss", choices=["gauss", "full", "gmm", "studentt"],
                   help="flow-matching SOURCE distribution, fitted on the standardized train "
                        "cloud. 'full' adds linear correlation between latent dims; 'gmm' adds "
                        "multimodality. FM only -- DDPM's source is fixed by its forward process.")
    p.add_argument("--gmm-k", type=int, default=8)
    p.add_argument("--nu", type=float, default=5.0, help="Student-t degrees of freedom")
    p.add_argument("--decode-weight", type=float, default=0.0,
                   help="exponent beta on per-latent DECODABILITY as a Stage 4 sampling weight. "
                        "0 = off (uniform). The flow then trains more on latents the frozen "
                        "decoder reconstructs well, so it puts less mass where decoding fails. "
                        "Targets the fm->ceiling gap. Implemented as a WeightedRandomSampler, so "
                        "the loss is untouched and the decoder still receives no gradient -- the "
                        "weights are precomputed constants.")
    p.add_argument("--sigma-min", type=float, default=0.0,
                   help="constant noise around the interpolant (stochastic interpolant). 0 is "
                        "the deterministic rectified-flow path. Non-zero widens the region of "
                        "x-space the field is trained on without changing the conditional "
                        "target, and is the standard answer to the diversity loss that OT "
                        "coupling causes by making the map more deterministic.")
    p.add_argument("--coupling", default="independent", choices=["independent", "ot"],
                   help="minibatch pairing of source to data. 'ot' is what makes a richer "
                        "--source worth having; see fm/model.py:ot_pair.")
    p.add_argument("--skip-stage5", action="store_true",
                   help="train and checkpoint only. Used for autoguidance guiding models, "
                        "whose standalone sample quality is not what they are for.")
    p.add_argument("--fm-warmup", type=int, default=500,
                   help="Stage 4 warmup steps. Scale with --fm-steps or a short run spends "
                        "most of training warming up and the schedule shape stops matching.")
    p.add_argument("--n-original", type=int, default=0,
                   help="with an extended corpus: keep the first N entries' splits fixed and "
                        "put everything after them in train, so valid/test/test_ion do not move")
    p.add_argument("--tokenizer", default="bpe", choices=["bpe", "selfies"],
                   help="decoder target vocabulary. 'selfies' makes every output a "
                        "valid molecule by construction, removing the parse-failure "
                        "term that is 74%% of the decoder ceiling loss (docs 6.17).")
    p.add_argument("--selfies-vocab", default="data/selfies_vocab.json")
    p.add_argument("--tag", default="run")
    p.add_argument("--out", default="runs/il/results")
    a = p.parse_args()

    dev = torch.device("cpu"); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(a.tokenizer, a.selfies_vocab)
    print(f"tokenizer {a.tokenizer}  vocab {tok.vocab_size}", flush=True)
    PAD, BOS = tok.pad_token_id, tok.bos_token_id

    def tokenize(smis):
        T = tok(smis, padding="max_length", truncation=True, max_length=a.max_len,
                return_tensors="pt")["input_ids"]
        return torch.cat([torch.full((len(T),1), BOS, dtype=torch.long), T[:, :-1]], 1), T

    stem = a.corpus
    ils = json.loads((DATA/f"{stem}.json").read_text())
    # il_pairs_v2 keeps the original unsuffixed embedding files; any later
    # corpus carries its version suffix (v3 = +CIR ILThermo, v4 = +OOD-in-train)
    esuf = "" if stem == "il_pairs_v2" else "_" + stem.rsplit("_", 1)[-1]
    E = torch.from_numpy(np.load(DATA/f"il_emb_ils{esuf}.npy")).float()
    S = (DATA/f"il_smi_ils{esuf}.txt").read_text().split("\n")
    assert len(E) == len(S) == len(ils)
    T_in, T = tokenize(S)

    ood = json.loads((DATA/f"{a.ood_stem}.json").read_text())
    osuf = "" if a.ood_stem == "il_ood_pairs" else "_" + a.ood_stem.rsplit("_", 1)[-1]
    E_ood = torch.from_numpy(np.load(DATA/f"il_emb_ood{osuf}.npy")).float()
    S_ood = [q["pair"] for q in ood]
    assert len(E_ood) == len(S_ood) == len(ood), f"{len(E_ood)} {len(S_ood)} {len(ood)}"
    Tin_ood, T_ood = tokenize(S_ood)

    sp = (make_splits_extended(ils, a.n_original, seed=a.split_seed) if a.n_original
          else make_splits(ils, seed=a.split_seed))
    print("splits: " + "  ".join(f"{k} {len(v)}" for k, v in sp.items())
          + f"  ood {len(ood)}", flush=True)

    model = LatentAE(E.shape[1], a.latent_dim, tok.vocab_size, a.max_len,
                     decoder=a.decoder, hidden=a.hidden, pad_id=PAD,
                     word_dropout=a.word_dropout).to(dev)
    ck = Path(a.pretrained)
    if ck.exists():
        sd = torch.load(ck, map_location=dev, weights_only=False)["model"]
        missing = model.load_state_dict(sd, strict=False)
        print(f"loaded pretrained decoder ({len(sd)} tensors); missing={len(missing.missing_keys)}", flush=True)
    else:
        print("WARNING: no pretrained checkpoint, training from scratch", flush=True)

    # ---- Stage 2: fit the autoencoder.
    #
    # Phases 2a and 2b are the SAME operation on different data -- the same
    # parameters, the same reconstruction loss, the same loop that Stage 1 runs.
    # "Pretrain" and "fine-tune" here are a data curriculum, not a change of
    # objective. What makes the LAST phase special is that Stage 3 freezes the
    # latent it produces, so whatever the autoencoder sees last decides the
    # geometry the flow has to model.
    #
    #   2a  ~100k enumerated train-ion pairs -- raises decoder capability
    #   2b  the 3,447 observed pairs, lower LR -- re-seats the latent on the
    #       real distribution, so Stage 4's target stays the real one
    #
    # Skipping 2b would hand the flow a latent organized for the uniform product
    # distribution of ions, which is the target-distribution error this whole
    # split is designed to avoid.
    Zfix = torch.randn(len(E), a.latent_dim) if a.latent_mode == "random" else None
    Zfix_ood = torch.randn(len(E_ood), a.latent_dim) if a.latent_mode == "random" else None
    tr_i = torch.from_numpy(sp["train"]).long(); va_i = torch.from_numpy(sp["valid"]).long()

    def run_phase(E_, Tin_, T_, epochs, lr, label, idx=None, report=5):
        dl = DataLoader(TensorDataset(E_, Tin_, T_, idx if idx is not None
                                      else torch.arange(len(E_))),
                        batch_size=a.batch_size, shuffle=True, drop_last=True)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs*max(len(dl),1))
        t0 = time.time(); acc = ex = 0.0
        for ep in range(epochs):
            model.train()
            for e,ti,to,ix in dl:
                zo = Zfix[ix].to(dev) if (Zfix is not None and idx is not None) else None
                o = model.loss(e.to(dev), ti.to(dev), to.to(dev), z_override=zo)
                opt.zero_grad(set_to_none=True); o["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            model.eval()
            acc, ex = recon_acc(model, E[va_i], T_in[va_i], T[va_i], dev,
                                None if Zfix is None else Zfix[va_i])
            if (ep+1) % report == 0 or ep == epochs-1:
                print(f"  {label} ep {ep+1}/{epochs}  val_token_acc {acc:.4f}  "
                      f"val_exact {ex:.4f}  {(time.time()-t0)/60:.1f}m", flush=True)
        return acc, ex

    if a.load_ae and Path(a.load_ae).exists():
        ck2 = torch.load(a.load_ae, map_location=dev, weights_only=False)
        model.load_state_dict(ck2["model"]); model.eval()
        acc, ex = ck2["val_token_acc"], ck2["val_exact"]
        print(f"loaded fine-tuned AE from {a.load_ae} (val_token_acc {acc:.4f}); "
              f"phases 2a/2b skipped", flush=True)
    else:
        if a.enum_pairs:
            Ee = torch.from_numpy(np.load(DATA/f"il_emb_{a.enum_pairs}.npy")).float()
            Se = (DATA/f"il_smi_{a.enum_pairs}.txt").read_text().split("\n")
            assert len(Ee) == len(Se), f"{len(Ee)} vs {len(Se)}"
            Tin_e, T_e = tokenize(Se)
            print(f"PHASE 2a: enumerated decoder corpus {len(Ee):,} pairs "
                  f"({len(Ee)/len(sp['train']):.0f}x the observed train split)", flush=True)
            run_phase(Ee, Tin_e, T_e, a.enum_epochs, a.enum_lr, "2a", idx=None, report=1)
        lbl = "2b" if a.enum_pairs else "ft"
        acc, ex = run_phase(E[tr_i], T_in[tr_i], T[tr_i], a.ft_epochs, a.ft_lr, lbl, idx=tr_i)

    gate_pass = acc >= a.gate
    print(f"STAGE 2 done: val_token_acc {acc:.4f}  gate({a.gate}) "
          f"{'PASS' if gate_pass else 'FAIL -- results below are suspect'}", flush=True)

    if a.save_ae:
        Path(a.save_ae).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "val_token_acc": acc, "val_exact": ex,
                    "args": vars(a)}, a.save_ae)
        print(f"saved fine-tuned AE -> {a.save_ae}", flush=True)

    # ---- per-split reconstruction: AE generalization, teacher-forced
    model.eval(); ae = {}
    for name, idx in sp.items():
        i = torch.from_numpy(idx).long()
        ta, te = recon_acc(model, E[i], T_in[i], T[i], dev, None if Zfix is None else Zfix[i])
        ae[name] = {"token_acc": ta, "exact": te}
    ta, te = recon_acc(model, E_ood, Tin_ood, T_ood, dev, Zfix_ood)
    ae["ood"] = {"token_acc": ta, "exact": te}
    print("STAGE 2 recon: " + "  ".join(f"{k} {v['token_acc']:.4f}" for k, v in ae.items()), flush=True)

    # ---- Stage 3: encode -> fixed z clouds, one per split
    with torch.no_grad():
        Zall = Zfix.clone() if Zfix is not None else torch.cat(
            [model.encode(E[i:i+512].to(dev)).cpu() for i in range(0, len(E), 512)])
        Zood = Zfix_ood.clone() if Zfix_ood is not None else torch.cat(
            [model.encode(E_ood[i:i+512].to(dev)).cpu() for i in range(0, len(E_ood), 512)])
    Z = {k: Zall[torch.from_numpy(v).long()] for k, v in sp.items()}; Z["ood"] = Zood

    # ---- optional: upweight a labelled sub-population, in the FLOW's cloud ONLY.
    #
    # §6.10 found the autoencoder absorbs an injected minority sub-population
    # immediately (OOD recon 0.924 -> 0.954, ceiling_ood -> 1.000) while the flow
    # barely moves. That points at density weighting rather than representation:
    # 150 points against 3,597 is 4% of the mass, and a velocity field fitted by
    # regression over the whole cloud will not spend capacity on a 4% mode.
    #
    # Replicating those rows here tests exactly that. Stage 2 has already run, so
    # the autoencoder and therefore the latent GEOMETRY are untouched -- pair this
    # with --load-ae and the sampling density is the single changed variable.
    Z_flow = Z["train"]; up_n = 0; up_k = int(round(a.upweight))
    if up_k > 1 and a.upweight_source:
        src = np.array([ils[i].get("source", "") == a.upweight_source for i in sp["train"]])
        up_n = int(src.sum())
        assert up_n, f"no train rows with source={a.upweight_source!r}"
        extra = torch.from_numpy(np.repeat(np.where(src)[0], up_k - 1)).long()
        Z_flow = torch.cat([Z["train"], Z["train"][extra]])
        print(f"UPWEIGHT: {up_n} rows tagged {a.upweight_source!r} at {up_k}x -> flow cloud "
              f"{len(Z['train'])} -> {len(Z_flow)} rows; that sub-population goes "
              f"{up_n/len(Z['train']):.1%} -> {up_n*up_k/len(Z_flow):.1%} of the density",
              flush=True)

    # ---- optional: SUBSET the flow's cloud, for an autoguidance guiding model.
    #
    # Autoguidance extrapolates away from a deliberately degraded version of the
    # same model: v~ = (1+w)*v1 - w*v0. That only cancels error if v0 fails in the
    # SAME direction as v1, just harder, which is why the weak model must differ
    # in data volume and nothing else.
    #
    # The Standardizer below is therefore fit on the FULL train cloud, never the
    # subset. Fit it on the subset and the two fields end up in different
    # coordinate frames, which makes the difference v1-v0 meaningless -- the
    # opposite of the --upweight case, where the frame only has to be
    # self-consistent within one model.
    sub_n = 0
    if a.train_subset_n and a.train_subset_n < len(Z_flow):
        assert up_n == 0, "--train-subset-n and --upweight are mutually exclusive"
        g = np.random.default_rng(a.train_subset_seed)
        keep = torch.from_numpy(np.sort(g.choice(len(Z_flow), a.train_subset_n,
                                                 replace=False))).long()
        sub_n = int(len(keep))
        Z_flow = Z["train"][keep]
        print(f"SUBSET: flow trains on {sub_n} of {len(Z['train'])} train latents "
              f"({sub_n/len(Z['train']):.1%}, seed {a.train_subset_seed}); "
              f"Standardizer still fit on all {len(Z['train'])}", flush=True)

    # ---- optional: numeric conditioning for classifier-free guidance.
    #
    # The property is standardized on the TRAIN-LABELLED rows only -- valid/test
    # labels never touch the scaler. Coverage is partial by nature (melting point
    # is reported for ~46% of the corpus), which maps cleanly onto CFG's own
    # mechanism: unlabelled rows are permanent null-token examples.
    y_val = y_msk = None
    if a.cond_prop:
        assert not sub_n and not up_n, "--cond-prop is exclusive with subset/upweight"
        field = {"mp": "melting_point_C"}[a.cond_prop]
        raw = np.array([float(q[field]) if q.get(field) not in (None, "") else np.nan
                        for q in ils], dtype=np.float64)
        tr = np.asarray(sp["train"])
        lab = np.isfinite(raw[tr])
        mu, sd = float(raw[tr][lab].mean()), float(raw[tr][lab].std())
        print(f"COND {a.cond_prop!r}: {lab.sum():,}/{len(tr):,} train rows labelled "
              f"({lab.mean():.1%});  mean {mu:.1f}  sd {sd:.1f}  "
              f"(scaler fit on train-labelled only)", flush=True)
        z = np.where(np.isfinite(raw), (raw - mu) / max(sd, 1e-8), 0.0)
        y_val = torch.from_numpy(z).float()
        y_msk = torch.from_numpy(np.isfinite(raw).astype(np.float32))
        Path(f"runs/il/_fm_{a.tag}").mkdir(parents=True, exist_ok=True)
        (Path(f"runs/il/_fm_{a.tag}") / "cond.json").write_text(json.dumps(
            {"prop": a.cond_prop, "field": field, "mean": mu, "std": sd,
             "n_labelled_train": int(lab.sum()), "cfg_dropout": a.cfg_dropout}))

    # fit on what the flow actually trains on -- still TRAIN only, no held-out leak.
    # Exception: a subset run standardizes against the full cloud (see above).
    sc = Standardizer().fit(Z["train"] if sub_n else Z_flow)
    print(f"STAGE 3: train z cloud {tuple(Z_flow.shape)}  per-dim std "
          f"[{Z_flow.std(0).min():.3f},{Z_flow.std(0).max():.3f}]", flush=True)

    # ---- Stage 4: flow matching on the TRAIN cloud (decoder untouched)
    # The valid cloud is passed only so the held-out FM loss gets logged. The
    # script samples from the final EMA, never from best.pt, so this monitoring
    # cannot influence which weights are used.
    src = None
    if a.source != "gauss":
        assert a.generative == "fm", "--source applies to flow matching only"
        src = make_source(a.source, k=a.gmm_k, nu=a.nu).fit(sc.transform(Z["train"]))
        extra = (f"  k={a.gmm_k}  loglik {src.ll:.4f}" if a.source == "gmm"
                 else f"  nu={a.nu}" if a.source == "studentt" else "")
        print(f"SOURCE: {a.source}{extra}  (fitted on the standardized train cloud)", flush=True)

    def build():
        net = make_velocity_net(a.fm_arch, a.latent_dim,
                                cond_dim=(2 if a.cond_prop else 0), width=a.fm_width)
        return (FlowMatching(net, path=CondOTPath(sigma_min=a.sigma_min), source=src,
                             coupling=a.coupling) if a.generative == "fm"
                else DDPM(net, T=a.ddpm_T, schedule=a.ddpm_schedule))

    def draw_source(n):
        return src.sample(n, device=dev) if src is not None else torch.randn(n, a.latent_dim)

    samplers = [s for s in (a.sampler or
                ("heun" if a.generative == "fm" else "ancestral")).split(",") if s]
    gen = build()
    cfg = TrainConfig(steps=a.fm_steps, lr=1e-3, out_dir=f"runs/il/_fm_{a.tag}",
                      device="cpu", seed=a.seed,
                      warmup_steps=min(a.fm_warmup, max(a.fm_steps//8, 1)),
                      log_every=max(a.fm_steps//4,1),
                      val_every=max(a.fm_steps//4,1), ckpt_every=0)
    if a.cond_prop:
        tr_i2 = torch.from_numpy(sp["train"]).long(); va_i2 = torch.from_numpy(sp["valid"]).long()
        tr_ds = CFGVectorDataset(sc.transform(Z_flow), y_val[tr_i2], y_msk[tr_i2],
                                 p_uncond=a.cfg_dropout, seed=a.split_seed)
        va_ds = CFGVectorDataset(sc.transform(Z["valid"]), y_val[va_i2], y_msk[va_i2],
                                 p_uncond=0.0, seed=a.split_seed)
    else:
        tr_ds = VectorDataset(sc.transform(Z_flow))
        va_ds = VectorDataset(sc.transform(Z["valid"]))
    sampler_kw = {"shuffle": True}
    if a.decode_weight > 0:
        # per-sample teacher-forced token accuracy on the TRAIN latents, computed once
        # with the frozen decoder. No gradient, no held-out data.
        accs = []
        with torch.no_grad():
            for i in range(0, len(tr_i), 256):
                ix = tr_i[i:i+256]
                zb = Zall[ix].to(dev)
                lg = model.dec(zb, T_in[ix].to(dev))
                tg = T[ix].to(dev)
                L = min(lg.size(1), tg.size(1))
                m = tg[:, :L] != PAD
                ok = (lg[:, :L].argmax(-1) == tg[:, :L]) & m
                accs.append((ok.sum(1) / m.sum(1).clamp_min(1)).cpu())
        dw_acc = torch.cat(accs).clamp_min(1e-3)
        w = dw_acc.pow(a.decode_weight)
        print(f"DECODE-WEIGHT: beta={a.decode_weight}  per-latent token acc "
              f"min {dw_acc.min():.3f} med {dw_acc.median():.3f} max {dw_acc.max():.3f}; "
              f"sampling weight ratio max/min {float(w.max()/w.min()):.1f}", flush=True)
        from torch.utils.data import WeightedRandomSampler
        sampler_kw = {"sampler": WeightedRandomSampler(w.double(), len(w), replacement=True)}

    trn = Trainer(gen, DataLoader(tr_ds, batch_size=128, drop_last=True, **sampler_kw), cfg,
                  val_loader=DataLoader(va_ds, batch_size=128),
                  has_cond=bool(a.cond_prop))
    t_tr = time.time(); trn.fit(); train_secs = time.time() - t_tr
    fm_val = trn.validate()
    ema = build()
    ema.load_state_dict(trn.ema.state_dict()); ema.to(dev).eval()

    if a.skip_stage5:
        print(f"STAGE 5 skipped (--skip-stage5); checkpoint at runs/il/_fm_{a.tag}/last.pt",
              flush=True)
        return

    # ---- Stage 5: generate + score
    n = a.n_samples
    trainP, trainC, trainA = ion_sets(ils, sp["train"])
    train_smi = [ils[i]["pair"] for i in sp["train"]]
    ELEM = element_set(train_smi); RH, RM = heavy_and_mw(train_smi)
    fwd = forward_flops(ema.net, batch=1)

    # arms that do not depend on the sampler: decoded once, reused for every row
    # the prior arm uses the SAME source, so it measures what the source alone
    # gives you and (fm - prior) stays the flow's own contribution
    shared = {"prior": sc.inverse(draw_source(n))}
    for k in ("train", "test", "test_ion"):
        shared[f"ceiling_{k}"] = Z[k][torch.randperm(len(Z[k]))[:n]]
    shared["ceiling_ood"] = Z["ood"]
    shared_m = {}
    for name, z in shared.items():
        sm = decode(model, z, tok, a.max_len, dev)
        m, pairs = plausibility(sm, len(z), ELEM, RH, RM)
        m |= novelty(pairs, trainP, trainC, trainA); m["validity"] = m["parses"]
        shared_m[name] = m
        (out/f"samples_{a.tag}_{name}.txt").write_text("\n".join(s for s in sm if s))
    for k in ("test", "test_ion"):
        smi = [ils[i]["pair"] for i in sp[k]]
        shared_m[f"real_{k}"], _ = plausibility(smi, len(smi), ELEM, RH, RM)
    shared_m["real_ood"], _ = plausibility([q["pair"] for q in ood], len(ood), ELEM, RH, RM)
    print("  shared arms: " + "  ".join(
        f"{k} {shared_m[k]['plausible']:.3f}" for k in
        ("prior", "ceiling_test", "ceiling_ood", "real_test")), flush=True)

    for sampler in samplers:
        tag = a.tag if len(samplers) == 1 else f"{a.tag}__{sampler}"
        t_s = time.time()
        if a.generative == "fm":
            raw = sample(ema, x0=draw_source(n), n_steps=a.ode_steps,
                         method=sampler, device=dev)
        else:
            raw = ema.sample(n, (a.latent_dim,), device=dev, sampler=sampler,
                             n_steps=a.ode_steps, clip_x0=a.clip_x0)
        sample_secs = time.time() - t_s
        gz = sc.inverse(raw.cpu())

        # both models pay exactly one forward pass per network evaluation, so the
        # whole efficiency difference is the NFE count
        nfe = n_function_evals(sampler, a.ddpm_T, a.ode_steps, sampler)
        cost = {"nfe_per_sample": nfe, "flops_fwd_per_eval": fwd,
                "flops_sample_per": sampling_flops(ema.net, nfe),
                "flops_train_total": train_flops(fwd, a.fm_steps * 128),
                "train_secs": train_secs, "sample_secs": sample_secs,
                "sample_secs_per_1k": sample_secs / max(n, 1) * 1000,
                "params": sum(q.numel() for q in ema.net.parameters())}

        res = {"schema": 2, "tag": tag, "latent_dim": a.latent_dim, "decoder": a.decoder,
               "latent_mode": a.latent_mode, "split_seed": a.split_seed,
               "generative": a.generative, "sampler": sampler, "ddpm_T": a.ddpm_T,
           "enum_pairs": a.enum_pairs, "enum_epochs": a.enum_epochs,
           "source": a.source, "coupling": a.coupling, "seed": a.seed,
           "sigma_min": a.sigma_min, "decode_weight": a.decode_weight,
           **({"fm_arch": a.fm_arch} if a.fm_arch != "mlp" else {}),
           **({"gmm_k": a.gmm_k} if a.source == "gmm" else {}),
           **({"nu": a.nu} if a.source == "studentt" else {}),
           "corpus": a.corpus, "ood_stem": a.ood_stem, "n_train": int(len(sp["train"])),
           "n_ood": len(ood),
           **({"upweight": up_k, "upweight_source": a.upweight_source,
               "n_upweighted": up_n, "n_flow_rows": int(len(Z_flow))} if up_n else {}),
           **({"train_subset_n": sub_n, "train_subset_seed": a.train_subset_seed,
               "fm_steps": a.fm_steps} if sub_n else {}),
               "ddpm_schedule": a.ddpm_schedule, "clip_x0": a.clip_x0, "cost": cost,
               "ft_val_token_acc": acc, "ft_val_exact": ex, "gate_pass": bool(gate_pass),
               "splits": {k: int(len(v)) for k, v in sp.items()} | {"ood": len(ood)},
               "ae_recon": ae, "fm_val_loss": fm_val,
               "sw2": {k: sliced_w2(gz.numpy(), Z[k].numpy()) for k in Z},
               "arms": dict(shared_m), "recovery": {}}

        sm = decode(model, gz, tok, a.max_len, dev)
        m, gen_pairs = plausibility(sm, n, ELEM, RH, RM)
        m |= novelty(gen_pairs, trainP, trainC, trainA); m["validity"] = m["parses"]
        res["arms"]["fm"] = m
        (out/f"samples_{tag}_fm.txt").write_text("\n".join(s for s in sm if s))

        _, C, A = ion_sets(ils, sp["test_ion"])
        res["recovery"]["test_ion"] = recovery(gen_pairs, C, A)
        res["recovery"]["ood"] = recovery(gen_pairs, {q["cation"] for q in ood},
                                          {q["anion"] for q in ood})
        with (out/"results.jsonl").open("a") as fh: fh.write(json.dumps(res)+"\n")
        print(f"  [{sampler}] plaus {m['plausible']:.3f}  uniq {m['uniqueness']:.3f}  "
              f"nov_cat {m['novel_cation_rate']:.3f}  sw2 test_ion {res['sw2']['test_ion']:.3f}  "
              f"ood {res['sw2']['ood']:.3f}  NFE {nfe}  sample {sample_secs:.0f}s  "
              f"train {train_secs:.0f}s", flush=True)
    print(f"STAGE 5 COMPLETE -> {out/'results.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
