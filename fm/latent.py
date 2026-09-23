"""A sequence autoencoder to give flow matching a well-behaved latent space.

Why this file exists, and the mistake it is here to prevent: it is tempting to
grab a strong pretrained encoder (ESM for proteins, ChemBERTa for molecules),
embed your data, and run flow matching on the embeddings. That does not work,
for two reasons.

  1. No decoder. ESM-style encoders map sequence -> vector. Nothing maps back.
     You would generate vectors you cannot turn into molecules or proteins, and
     therefore cannot evaluate at all.

  2. No regularization. Even with a decoder, a plain autoencoder's latent space
     has arbitrary geometry: the encoder scatters data wherever is convenient,
     leaving large regions that decode to nonsense. Flow matching will happily
     generate points in those regions, because it only ever saw where the data
     *is*, never where the decoder is *valid*.

So the latent has to be trained, with a decoder, and regularized. This is a VAE
with a deliberately small KL weight -- enough to keep the latent smooth and
roughly Gaussian, not so much that reconstruction collapses. Tuning `beta` and
watching what it does to reconstruction accuracy versus sample validity is the
single most instructive experiment in this track.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SeqVAE(nn.Module):
    """GRU encoder/decoder over a token sequence, with a fixed-size latent.

    Args:
        vocab_size: number of tokens including PAD/BOS/EOS.
        latent_dim: dimension of the space flow matching will operate in.
        emb_dim / hidden: sizes of the token embedding and GRU state.
        beta: KL weight. 0 gives a plain autoencoder (sharper reconstruction,
            worse latent for generation); ~1e-3 to 1e-2 is a sensible range.
    """

    def __init__(
        self,
        vocab_size: int,
        latent_dim: int = 64,
        emb_dim: int = 128,
        hidden: int = 384,
        n_layers: int = 1,
        beta: float = 3e-3,
        pad_id: int = 0,
        word_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.beta = beta
        self.pad_id = pad_id
        self.word_dropout = word_dropout

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.enc = nn.GRU(emb_dim, hidden, n_layers, batch_first=True, bidirectional=True)
        self.to_mu = nn.Linear(2 * hidden, latent_dim)
        self.to_logvar = nn.Linear(2 * hidden, latent_dim)

        self.from_z = nn.Linear(latent_dim, hidden * n_layers)
        self.dec = nn.GRU(emb_dim + latent_dim, hidden, n_layers, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)
        self.n_layers, self.hidden = n_layers, hidden

    def encode(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        h, _ = self.enc(self.emb(tokens))
        mask = (tokens != self.pad_id).float().unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)  # mean over real tokens
        return self.to_mu(pooled), self.to_logvar(pooled).clamp(-8, 8)

    def decode(self, z: Tensor, tokens_in: Tensor) -> Tensor:
        """Teacher-forced decode. tokens_in is the sequence shifted right (BOS-prefixed)."""
        h0 = self.from_z(z).view(-1, self.n_layers, self.hidden).transpose(0, 1).contiguous()
        e = self.emb(tokens_in)
        # z is concatenated at every step so the decoder cannot forget it.
        e = torch.cat([e, z.unsqueeze(1).expand(-1, e.size(1), -1)], dim=-1)
        h, _ = self.dec(e, h0)
        return self.out(h)

    def forward(self, tokens: Tensor, tokens_in: Tensor, tokens_out: Tensor) -> dict:
        if self.training and self.word_dropout > 0:
            # Randomly blank the decoder's own previous tokens. This is the standard
            # fix for posterior collapse (Bowman et al. 2016): an autoregressive
            # decoder will ignore z whenever the previous tokens alone predict the
            # next one, so we take those away and force it to consult z.
            drop = torch.rand_like(tokens_in, dtype=torch.float) < self.word_dropout
            drop[:, 0] = False  # always keep BOS
            tokens_in = tokens_in.masked_fill(drop, self.pad_id)
        mu, logvar = self.encode(tokens)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        logits = self.decode(z, tokens_in)
        recon = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), tokens_out.reshape(-1),
            ignore_index=self.pad_id,
        )
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)).mean()
        with torch.no_grad():
            valid = tokens_out != self.pad_id
            acc = ((logits.argmax(-1) == tokens_out) & valid).sum() / valid.sum().clamp_min(1)
        return {"loss": recon + self.beta * kl, "recon": recon, "kl": kl, "token_acc": acc}

    @torch.no_grad()
    def generate(self, z: Tensor, max_len: int, bos_id: int, eos_id: int) -> Tensor:
        """Greedy autoregressive decode from latents."""
        B = z.size(0)
        h = self.from_z(z).view(B, self.n_layers, self.hidden).transpose(0, 1).contiguous()
        tok = torch.full((B, 1), bos_id, dtype=torch.long, device=z.device)
        done = torch.zeros(B, dtype=torch.bool, device=z.device)
        outs = []
        for _ in range(max_len):
            e = torch.cat([self.emb(tok), z.unsqueeze(1)], dim=-1)
            o, h = self.dec(e, h)
            tok = self.out(o[:, -1]).argmax(-1, keepdim=True)
            tok = torch.where(done.unsqueeze(1), torch.full_like(tok, self.pad_id), tok)
            done = done | (tok.squeeze(1) == eos_id)
            outs.append(tok)
            if bool(done.all()):
                break
        return torch.cat(outs, dim=1)
