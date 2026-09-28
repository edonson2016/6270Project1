"""SELFIES tokenizer with the subset of the HF tokenizer interface this pipeline uses.

Motivation (docs 6.17): 74% of the decoder ceiling's loss is SMILES PARSE
failure -- 7.4 of 9.7 points. SELFIES strings decode to a valid molecule by
construction, so that entire term goes to zero.

The risk is that it converts a detectable failure into an undetectable one: an
unparseable SMILES is rejected, whereas a wrong SELFIES is a valid molecule that
may silently be one fragment or charge-unbalanced. Matched-rate corruption says
SELFIES degrades far more gracefully anyway (charge balance 0.474 vs 0.097 at 5%
token corruption), but random substitution is a poor model of a trained decoder's
errors, so the real test is the ceiling arm itself.

Duck-types the attributes and calls that il_pretrain.py / il_finetune_fm.py use,
so swapping is a one-line change at each construction site.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


class SelfiesTokenizer:
    PAD, BOS, EOS, UNK = "[pad]", "[bos]", "[eos]", "[unk]"

    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_token_id = self.stoi[self.PAD]
        self.bos_token_id = self.stoi[self.BOS]
        self.eos_token_id = self.stoi[self.EOS]
        self.unk_token_id = self.stoi[self.UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # ---- construction
    @classmethod
    def build(cls, smiles_iter) -> "SelfiesTokenizer":
        import selfies as sf
        alpha = set()
        for s in smiles_iter:
            try:
                alpha.update(sf.split_selfies(sf.encoder(s)))
            except Exception:
                continue
        return cls([cls.PAD, cls.BOS, cls.EOS, cls.UNK] + sorted(alpha))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"itos": self.itos}))

    @classmethod
    def load(cls, path) -> "SelfiesTokenizer":
        return cls(json.loads(Path(path).read_text())["itos"])

    # ---- HF-compatible surface
    def __call__(self, smiles, padding="max_length", truncation=True,
                 max_length=80, return_tensors="pt"):
        import selfies as sf
        if isinstance(smiles, str):
            smiles = [smiles]
        out = torch.full((len(smiles), max_length), self.pad_token_id, dtype=torch.long)
        for r, s in enumerate(smiles):
            try:
                toks = list(sf.split_selfies(sf.encoder(s)))
            except Exception:
                toks = []
            ids = [self.stoi.get(t, self.unk_token_id) for t in toks][: max_length - 1]
            ids.append(self.eos_token_id)
            out[r, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": out}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        """ids -> SELFIES -> SMILES. Returns '' when SELFIES itself rejects the string."""
        import selfies as sf
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        toks = [self.itos[int(i)] for i in ids
                if not (skip_special_tokens and int(i) in special)]
        toks = [t for t in toks if t != self.UNK]
        try:
            return sf.decoder("".join(toks)) or ""
        except Exception:
            return ""
