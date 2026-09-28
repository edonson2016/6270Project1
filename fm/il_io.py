"""Shared loaders for the IL scripts: tokenizer, autoencoder, latents, decoding, results."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .il_eval import element_set, heavy_and_mw, ion_sets, repair_radicals
from .il_pipeline import LatentAE
from .model import FlowMatching
from .nets import VelocityMLP

DATA = Path("data")
MODEL = "seyonec/ChemBERTa-zinc-base-v1"
RESULTS = Path("runs/il/results")


def load_tokenizer(kind: str = "bpe", selfies_vocab: str = "data/selfies_vocab.json"):
    if kind == "selfies":
        from .selfies_tok import SelfiesTokenizer
        return SelfiesTokenizer.load(selfies_vocab)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def load_ae(path, latent_dim: int, tok, max_len: int, dev="cpu", decoder: str = "ar") -> LatentAE:
    ae = LatentAE(768, latent_dim, tok.vocab_size, max_len, decoder=decoder, hidden=512,
                  pad_id=tok.pad_token_id)
    ae.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["model"])
    return ae.eval()


def load_ils(name: str = "il_pairs_v2") -> list[dict]:
    return json.loads((DATA / f"{name}.json").read_text())


def load_emb(name: str = "il_emb_ils.npy") -> Tensor:
    return torch.from_numpy(np.load(DATA / name)).float()


@torch.no_grad()
def encode_all(ae: LatentAE, E: Tensor, bs: int = 512) -> Tensor:
    return torch.cat([ae.encode(E[i:i + bs]) for i in range(0, len(E), bs)])


@torch.no_grad()
def decode(ae: LatentAE, z: Tensor, tok, max_len: int, dev="cpu", bs: int = 256,
           repair: bool = True) -> list[str]:
    """Greedy-decode latents (raw, unstandardized frame) to SMILES."""
    special = (tok.pad_token_id, tok.bos_token_id, tok.eos_token_id)
    out = []
    for i in range(0, len(z), bs):
        t = ae.dec.generate(z[i:i + bs].to(dev), max_len, tok.bos_token_id, tok.eos_token_id)
        for row in t.cpu().numpy():
            ids = [int(x) for x in row if int(x) not in special]
            s = tok.decode(ids, skip_special_tokens=True).replace(" ", "")
            out.append(repair_radicals(s) if repair else s)
    return out


def train_reference(ils: list[dict], train_idx):
    """(trainP, trainC, trainA, ELEM, RH, RM): the train-set context the scorers need."""
    trainP, trainC, trainA = ion_sets(ils, train_idx)
    smi = [ils[i]["pair"] for i in train_idx]
    return (trainP, trainC, trainA, element_set(smi), *heavy_and_mw(smi))


def load_results() -> dict[str, dict]:
    """results.jsonl rows keyed by tag (a later row for the same tag wins)."""
    return {r["tag"]: r for r in (json.loads(l) for l in open(RESULTS / "results.jsonl"))}


def load_cond(tag: str, ils: list[dict]):
    """(mean, std, raw label per IL with NaN where missing) for a conditional run."""
    cond = json.loads((Path(f"runs/il/_fm_{tag}") / "cond.json").read_text())
    raw = np.array([float(q[cond["field"]]) if q.get(cond["field"]) not in (None, "") else np.nan
                    for q in ils], dtype=np.float64)
    return cond["mean"], cond["std"], raw


def load_cfg_fm(tag: str, latent_dim: int, dev="cpu") -> FlowMatching:
    """EMA weights of a CFG-trained (cond_dim=2) flow."""
    fm = FlowMatching(VelocityMLP(latent_dim, cond_dim=2, width=384, depth=4))
    fm.load_state_dict(torch.load(f"runs/il/_fm_{tag}/last.pt", map_location=dev,
                                  weights_only=False)["ema"])
    return fm.eval()


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))
