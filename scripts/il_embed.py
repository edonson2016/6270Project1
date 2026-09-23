"""Stage 0: embed every corpus once with the frozen encoder, cache to disk.

The encoder never trains and never runs again after this, so paying for it once
and caching is strictly better than keeping it in any training loop. Everything
downstream reads .npy files.

Corpora:
  moses  SMILES grammar for decoder pretraining
  ions   the IL ion inventory -- puts charges and bracket atoms into pretraining
  ils    the 4,790 IL pairs, the actual target distribution
  ood    the 227 ILThermo/PubChem pairs absent from the Zenodo corpus -- a
         held-out test set from a different curation pipeline, never trained on
  enum   ~100k cation.anion combinations built from TRAIN ions only. Decoder
         training data; the flow must never see it (see build_il_enumerated.py)
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np, torch

MODEL = "seyonec/ChemBERTa-zinc-base-v1"
DATA = Path("data")


def load_moses(n: int, seed: int = 0) -> list[str]:
    smi, split = [], []
    with (DATA / "moses.csv").open() as f:
        for row in csv.DictReader(f):
            smi.append(row["SMILES"]); split.append(row.get("SPLIT", "train"))
    smi = np.array(smi)[np.array(split) == "train"]
    if n and n < len(smi):
        smi = smi[np.random.default_rng(seed).choice(len(smi), n, replace=False)]
    return list(smi)


def load_ions() -> list[str]:
    out = []
    for f in ("il_cations.csv", "il_anions.csv"):
        with (DATA / f).open() as fh:
            for row in csv.DictReader(fh):
                s = row.get("smiles", "").strip()
                if s:
                    out.append(s)
    return out


def load_ils() -> list[str]:
    return [p["pair"] for p in json.loads((DATA / "il_pairs_v2.json").read_text())]


def load_enum() -> list[str]:
    """Enumerated train-ion pairs for decoder training (scripts/build_il_enumerated.py)."""
    return (DATA / "il_smi_enum.txt").read_text().split("\n")


def load_ood() -> list[str]:
    """The ILThermo/PubChem held-out set (scripts/build_il_ood.py)."""
    return [p["pair"] for p in json.loads((DATA / "il_ood_pairs.json").read_text())]


def embed(smiles: list[str], tag: str, batch: int, max_len: int) -> None:
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = AutoModel.from_pretrained(MODEL).eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    out_e = DATA / f"il_emb_{tag}.npy"
    out_s = DATA / f"il_smi_{tag}.txt"
    embs, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(smiles), batch):
            b = tok(smiles[i:i+batch], padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
            h = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            embs.append(((h * m).sum(1) / m.sum(1).clamp_min(1)).cpu())
            done = i + batch
            if done % (batch * 40) == 0:
                el = time.time() - t0
                print(f"  [{tag}] {min(done,len(smiles)):>7d}/{len(smiles)}  "
                      f"{done/el:.0f}/s  eta {(len(smiles)-done)/max(done/el,1e-9)/60:.0f} min",
                      flush=True)
    E = torch.cat(embs).numpy().astype(np.float32)
    np.save(out_e, E)
    out_s.write_text("\n".join(smiles))
    print(f"  [{tag}] wrote {out_e} {E.shape} in {(time.time()-t0)/60:.1f} min", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--moses-n", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--only", default="", help="comma list: moses,ions,ils,ood,enum")
    a = p.parse_args()
    only = set(a.only.split(",")) if a.only else {"ils", "ions", "moses", "ood"}

    # smallest first, so a crash still leaves the critical corpus done
    if "ood" in only:
        s = load_ood();  print(f"ood   : {len(s)}", flush=True);  embed(s, "ood", a.batch, a.max_len)
    if "enum" in only:
        s = load_enum(); print(f"enum  : {len(s)}", flush=True); embed(s, "enum", a.batch, a.max_len)
    if "ils" in only:
        s = load_ils();  print(f"ils   : {len(s)}", flush=True);  embed(s, "ils", a.batch, a.max_len)
    if "ions" in only:
        s = load_ions(); print(f"ions  : {len(s)}", flush=True); embed(s, "ions", a.batch, a.max_len)
    if "moses" in only:
        s = load_moses(a.moses_n); print(f"moses : {len(s)}", flush=True)
        embed(s, "moses", a.batch, a.max_len)
    print("STAGE 0 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
