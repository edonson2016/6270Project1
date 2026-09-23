"""Cheap go/no-go: what does a frozen ChemBERTa actually give us on IL pairs?

Answers three things before any decoder is trained:
  1. does the PCA spectrum have an elbow, or is it a smooth decay?
  2. how many components to reach a given variance level?
  3. is variance even the right criterion -- do low-variance directions carry
     information the decoder would need (e.g. which ion is which)?
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

MODEL = "DeepChem/ChemBERTa-77M-MLM"


def main():
    pairs = json.loads(Path("data/il_pairs_v2.json").read_text())
    smiles = [p["pair"] for p in pairs]
    print(f"{len(smiles)} IL pairs")

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = AutoModel.from_pretrained(MODEL).eval()
    print(f"encoder: {MODEL}  hidden={enc.config.hidden_size}  "
          f"params={sum(p.numel() for p in enc.parameters())/1e6:.1f}M")

    embs = []
    with torch.no_grad():
        for i in range(0, len(smiles), 64):
            b = tok(smiles[i:i+64], padding=True, truncation=True,
                    max_length=256, return_tensors="pt")
            h = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            embs.append(((h * m).sum(1) / m.sum(1)).cpu())   # mean-pool over real tokens
    X = torch.cat(embs).numpy().astype(np.float64)
    np.save("data/il_chemberta_emb.npy", X.astype(np.float32))
    print(f"embeddings {X.shape}")

    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2 / (S**2).sum()
    cum = np.cumsum(var)
    print("\nPCA spectrum")
    print(f"  {'d':>5s}{'cum. variance':>16s}")
    for d in (1, 2, 4, 8, 16, 32, 64, 128, 256, min(384, len(var))):
        if d <= len(var):
            print(f"  {d:>5d}{cum[d-1]:>16.4f}")
    for t in (0.90, 0.95, 0.99, 0.999):
        print(f"  components for {t:.1%} variance: {int(np.searchsorted(cum, t))+1}")

    # participation ratio: a scale-free "effective number of directions"
    pr = (S**2).sum()**2 / (S**4).sum()
    print(f"\n  participation ratio (effective rank): {pr:.1f} of {len(var)}")

    # elbow test: on a log-log plot a power law is a straight line (no elbow).
    k = np.arange(1, min(200, len(var)) + 1)
    lv = np.log(var[:len(k)]); lk = np.log(k)
    slope, icept = np.polyfit(lk, lv, 1)
    resid = lv - (slope * lk + icept)
    print(f"  log-log power-law fit: slope {slope:.2f}, residual std {resid.std():.3f}")
    print("  (small residual = smooth power law = NO elbow; a real elbow leaves a big residual)")

    # does variance rank == information rank? check how well each PC separates anions
    an = np.array([p["anion"] for p in pairs])
    top = [a for a, _ in sorted({a: (an == a).sum() for a in set(an)}.items(),
                                key=lambda x: -x[1])[:8]]
    mask = np.isin(an, top)
    Z = Xc @ Vt.T
    f = []
    for d in range(min(64, Z.shape[1])):
        z, lab = Z[mask, d], an[mask]
        gm = z.mean()
        between = sum((lab == a).sum() * (z[lab == a].mean() - gm) ** 2 for a in top)
        within = sum(((z[lab == a] - z[lab == a].mean()) ** 2).sum() for a in top)
        f.append(between / max(within, 1e-9))
    f = np.array(f)
    print(f"\n  anion-discriminating power by PC index (F-ratio, top-8 anions):")
    print(f"    PCs  1-8 : {f[:8].mean():.4f}")
    print(f"    PCs  9-16: {f[8:16].mean():.4f}")
    print(f"    PCs 17-32: {f[16:32].mean():.4f}")
    print(f"    PCs 33-64: {f[32:64].mean():.4f}")
    print(f"    best single PC for anion identity: #{int(f.argmax())+1} (F={f.max():.3f})")


if __name__ == "__main__":
    main()
