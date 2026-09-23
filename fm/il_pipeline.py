"""Components for the ionic-liquid latent flow matching pipeline.

Architecture (see docs/IL_EXPERIMENT.md for the full rationale):

    SMILES -> [frozen ChemBERTa] -> mean-pool e (768)
           -> MLP_down -> z (d)           <- flow matching lives HERE
           -> MLP_up   -> u
           -> decoder  -> SMILES

Only MLP_down / MLP_up / decoder train here. The encoder is frozen and is run
once offline (scripts/il_embed.py), so nothing in this file ever touches it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def mlp(d_in: int, d_out: int, width: int, depth: int) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    layers += [nn.Linear(d, d_out)]
    return nn.Sequential(*layers)


class ARDecoder(nn.Module):
    """Autoregressive GRU decoder conditioned on z.

    z enters twice on purpose: once as the initial hidden state, and again
    concatenated to every input token. The second path matters because a GRU
    will otherwise wash the initial state out over a long sequence.
    """

    def __init__(self, vocab: int, d_latent: int, emb_dim: int = 128,
                 hidden: int = 512, n_layers: int = 1, pad_id: int = 1,
                 word_dropout: float = 0.25) -> None:
        super().__init__()
        self.vocab, self.pad_id, self.word_dropout = vocab, pad_id, word_dropout
        self.hidden, self.n_layers = hidden, n_layers
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=pad_id)
        self.up = mlp(d_latent, hidden * n_layers, 256, 2)
        self.rnn = nn.GRU(emb_dim + d_latent, hidden, n_layers, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def _h0(self, z: Tensor) -> Tensor:
        return self.up(z).view(-1, self.n_layers, self.hidden).transpose(0, 1).contiguous()

    def forward(self, z: Tensor, tokens_in: Tensor) -> Tensor:
        if self.training and self.word_dropout > 0:
            # Blank the decoder's own previous tokens so it cannot succeed by
            # language-modelling alone and must consult z. Guards the
            # teacher-forcing shortcut (see docs).
            drop = torch.rand_like(tokens_in, dtype=torch.float) < self.word_dropout
            drop[:, 0] = False
            tokens_in = tokens_in.masked_fill(drop, self.pad_id)
        e = self.emb(tokens_in)
        e = torch.cat([e, z.unsqueeze(1).expand(-1, e.size(1), -1)], -1)
        h, _ = self.rnn(e, self._h0(z))
        return self.out(h)

    @torch.no_grad()
    def generate(self, z: Tensor, max_len: int, bos_id: int, eos_id: int) -> Tensor:
        B = z.size(0)
        h = self._h0(z)
        tok = torch.full((B, 1), bos_id, dtype=torch.long, device=z.device)
        done = torch.zeros(B, dtype=torch.bool, device=z.device)
        outs = []
        for _ in range(max_len):
            e = torch.cat([self.emb(tok), z.unsqueeze(1)], -1)
            o, h = self.rnn(e, h)
            tok = self.out(o[:, -1]).argmax(-1, keepdim=True)
            tok = torch.where(done.unsqueeze(1), torch.full_like(tok, self.pad_id), tok)
            done = done | (tok.squeeze(1) == eos_id)
            outs.append(tok)
            if bool(done.all()):
                break
        return torch.cat(outs, 1)


class NARDecoder(nn.Module):
    """Non-autoregressive decoder: z expands to L position vectors, decoded in parallel.

    There is no left-to-right path, so there is no teacher-forcing shortcut to
    exploit and word dropout is unnecessary. The cost is that z must carry the
    whole molecule: no position can lean on its neighbours.
    """

    def __init__(self, vocab: int, d_latent: int, max_len: int,
                 hidden: int = 512, pad_id: int = 1) -> None:
        super().__init__()
        self.vocab, self.max_len, self.pad_id = vocab, max_len, pad_id
        self.up = mlp(d_latent, hidden, 512, 2)
        self.pos = nn.Parameter(torch.randn(max_len, hidden) * 0.02)
        self.body = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.out = nn.Linear(hidden, vocab)

    def forward(self, z: Tensor, tokens_in: Tensor | None = None) -> Tensor:
        u = self.up(z).unsqueeze(1) + self.pos.unsqueeze(0)   # (B, L, H)
        return self.out(self.body(u))

    @torch.no_grad()
    def generate(self, z: Tensor, max_len: int, bos_id: int, eos_id: int) -> Tensor:
        return self.forward(z).argmax(-1)


class LatentAE(nn.Module):
    """MLP_down + decoder. The frozen encoder's output `e` is the input."""

    def __init__(self, emb_dim: int, d_latent: int, vocab: int, max_len: int,
                 decoder: str = "ar", down_width: int = 512, down_depth: int = 2,
                 hidden: int = 512, pad_id: int = 1, word_dropout: float = 0.25) -> None:
        super().__init__()
        self.d_latent, self.decoder_kind, self.pad_id = d_latent, decoder, pad_id
        self.down = mlp(emb_dim, d_latent, down_width, down_depth)
        if decoder == "ar":
            self.dec = ARDecoder(vocab, d_latent, hidden=hidden, pad_id=pad_id,
                                 word_dropout=word_dropout)
        elif decoder == "nar":
            self.dec = NARDecoder(vocab, d_latent, max_len, hidden=hidden, pad_id=pad_id)
        else:
            raise ValueError(decoder)

    def encode(self, e: Tensor) -> Tensor:
        return self.down(e)

    def loss(self, e: Tensor, tokens_in: Tensor, tokens_out: Tensor,
             z_override: Tensor | None = None) -> dict:
        """z_override supplies z directly, bypassing MLP_down.

        This is how the random-z control arm runs: the same decoder is trained
        against latents that carry no chemical organization at all.
        """
        z = self.encode(e) if z_override is None else z_override
        logits = self.dec(z, tokens_in)
        L = min(logits.size(1), tokens_out.size(1))
        logits, tgt = logits[:, :L], tokens_out[:, :L]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tgt.reshape(-1), ignore_index=self.pad_id)
        with torch.no_grad():
            m = tgt != self.pad_id
            acc = ((logits.argmax(-1) == tgt) & m).sum() / m.sum().clamp_min(1)
            exact = (((logits.argmax(-1) == tgt) | ~m).all(1)).float().mean()
        return {"loss": loss, "token_acc": acc, "exact": exact, "z": z}
