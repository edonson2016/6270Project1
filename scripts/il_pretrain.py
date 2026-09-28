"""Stage 1: pretrain the decoder as a full autoencoder on MOSES + the IL ions.

Deliberately NOT plain language-model pretraining. The decoder learns `z -> SMILES`
as its actual task from the start, so it never acquires the habit of writing
plausible molecules without consulting z -- which is exactly the habit that would
make the latent useless later.

Including the IL ion inventory means charges and bracket atoms appear during
pretraining, so the IL fine-tune is not introducing them cold.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm.il_pipeline import LatentAE
from fm.il_io import DATA, load_tokenizer


def tokenize(smiles, tok, max_len):
    b = tok(smiles, padding="max_length", truncation=True, max_length=max_len,
            return_tensors="pt")
    return b["input_ids"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--decoder", default="ar", choices=["ar", "nar"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--max-len", type=int, default=80)
    p.add_argument("--word-dropout", type=float, default=0.25)
    p.add_argument("--tokenizer", default="bpe", choices=["bpe", "selfies"],
                   help="decoder target vocabulary. 'selfies' makes every output a "
                        "valid molecule by construction, removing the parse-failure "
                        "term that is 74%% of the decoder ceiling loss (docs 6.17).")
    p.add_argument("--selfies-vocab", default="data/selfies_vocab.json")
    p.add_argument("--out", default="runs/il/pretrain")
    a = p.parse_args()

    dev = torch.device("cpu"); torch.manual_seed(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(a.tokenizer, a.selfies_vocab)
    print(f"tokenizer {a.tokenizer}  vocab {tok.vocab_size}", flush=True)
    PAD, BOS, EOS = tok.pad_token_id, tok.bos_token_id, tok.eos_token_id

    E, S = [], []
    for tag in __import__("os").environ.get("PRETRAIN_TAGS","moses,ions").split(","):
        f = DATA / f"il_emb_{tag}.npy"
        if not f.exists():
            print(f"  (missing {f}, skipping {tag})"); continue
        E.append(np.load(f))
        S += (DATA / f"il_smi_{tag}.txt").read_text().split("\n")
    E = torch.from_numpy(np.concatenate(E)).float()
    print(f"pretrain corpus: {len(E):,} molecules, emb dim {E.shape[1]}", flush=True)
    assert len(E) == len(S), f"embedding/SMILES mismatch {len(E)} vs {len(S)}"

    T = tokenize(S, tok, a.max_len)
    T_in = torch.cat([torch.full((len(T), 1), BOS, dtype=torch.long), T[:, :-1]], 1)

    ds = TensorDataset(E, T_in, T)
    nval = min(4000, len(ds)//20)
    tr, va = random_split(ds, [len(ds)-nval, nval], generator=torch.Generator().manual_seed(0))
    dl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, drop_last=True)
    dv = DataLoader(va, batch_size=512)

    model = LatentAE(E.shape[1], a.latent_dim, tok.vocab_size, a.max_len,
                     decoder=a.decoder, hidden=a.hidden, pad_id=PAD,
                     word_dropout=a.word_dropout).to(dev)
    print(f"model: {sum(q.numel() for q in model.parameters())/1e6:.2f}M params  "
          f"decoder={a.decoder} d={a.latent_dim}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs*len(dl))

    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); agg = {"loss":0.,"token_acc":0.,"n":0}
        for i,(e,ti,to) in enumerate(dl):
            o = model.loss(e.to(dev), ti.to(dev), to.to(dev))
            opt.zero_grad(set_to_none=True); o["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            agg["loss"] += float(o["loss"].detach()); agg["token_acc"] += float(o["token_acc"]); agg["n"] += 1
            if (i+1) % 100 == 0:
                print(f"  ep{ep+1} step {i+1}/{len(dl)}  loss {agg['loss']/agg['n']:.4f}  "
                      f"acc {agg['token_acc']/agg['n']:.4f}  {(time.time()-t0)/60:.1f}m", flush=True)
        model.eval(); va_acc=va_ex=n=0
        with torch.no_grad():
            for e,ti,to in dv:
                o = model.loss(e.to(dev), ti.to(dev), to.to(dev))
                va_acc += float(o["token_acc"]); va_ex += float(o["exact"]); n += 1
        print(f"EPOCH {ep+1}/{a.epochs}  train_acc {agg['token_acc']/agg['n']:.4f}  "
              f"val_acc {va_acc/n:.4f}  val_exact {va_ex/n:.4f}  {(time.time()-t0)/60:.1f}m", flush=True)
        torch.save({"model": model.state_dict(), "args": vars(a),
                    "emb_dim": E.shape[1], "vocab": tok.vocab_size}, out/"pretrained.pt")
    print(f"STAGE 1 COMPLETE -> {out/'pretrained.pt'}", flush=True)


if __name__ == "__main__":
    main()
