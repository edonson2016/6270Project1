"""Oracle-free test of melting-point conditioning.

Every melting-point result so far has been read through a learned predictor, and
a predictor can be miscalibrated (the structure oracle turned out to carry a
0.750 slope) or fooled by a spurious correlation it shares with the flow. This
test uses NO predictor at all.

Generate at a requested melting point T. Take the REAL held-out ILs whose
MEASURED melting point falls in band B. Compare the two latent clouds with
sliced Wasserstein. Do it for every (T, B) pair.

  If conditioning works, each column's minimum sits on the diagonal:
  the cloud generated at 80 C should sit closer to real ILs measured at
  80 C than any other request does.

Read COLUMNS, not rows. Within a column the reference cloud is fixed, so the
numbers are directly comparable and band size cancels. Across a row the
reference changes size (43-88 molecules here) and sliced-W2 is biased upward on
small references, so row comparisons would be reading finite-sample inflation --
the same trap that made an OOD result look like a regression in 6.10.

A real-vs-real control matrix runs first. It compares train-labelled bands
against the same held-out bands, establishing whether these bands are separable
in this latent space AT ALL. If the control has no diagonal, the latent does not
encode melting point and nothing about the generated matrix is interpretable.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm import CFGVelocity, Standardizer, sample
from fm.il_eval import make_splits, sliced_w2
from fm.il_io import (encode_all, load_ae, load_cfg_fm, load_cond, load_emb, load_ils,
                      load_tokenizer, write_json)

EDGES = [-100, 20, 60, 100, 140, 180, 400]


def show(mat, rows, cols, title, rowlab):
    print(f"\n{title}")
    hdr = f"{rowlab:>14} " + "".join(f"{c:>10}" for c in cols)
    print(hdr); print("-" * len(hdr))
    best = np.nanargmin(mat, axis=0)
    for i, r in enumerate(rows):
        cells = "".join(
            (f"{mat[i,j]:>9.3f}" + ("*" if best[j] == i else " ")) for j in range(mat.shape[1]))
        print(f"{r:>14} {cells}")
    ok = sum(1 for j in range(mat.shape[1]) if best[j] == j)
    print(f"{'':>14} column minima on the diagonal: {ok}/{mat.shape[1]}"
          f"   (* = column minimum)")
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="cfg_mp_d16")
    p.add_argument("--ae", default="runs/il/ae/ae_enum_d16.pt")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--weights", default="0,1,2")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--tokenizer", default="bpe", choices=["bpe", "selfies"])
    p.add_argument("--max-len", type=int, default=80)
    p.add_argument("--selfies-vocab", default="data/selfies_vocab.json")
    p.add_argument("--emb", default="il_emb_ils.npy")
    p.add_argument("--out", default="runs/il/results/band_match.json")
    a = p.parse_args()

    dev = torch.device("cpu"); torch.manual_seed(0)
    tok = load_tokenizer(a.tokenizer, a.selfies_vocab)
    ae = load_ae(a.ae, a.latent_dim, tok, a.max_len, dev)

    ils = load_ils()
    sp = make_splits(ils, seed=0)
    mu, sd, raw = load_cond(a.tag, ils)
    Zall = encode_all(ae, load_emb(a.emb))
    sc = Standardizer().fit(Zall[torch.from_numpy(sp["train"]).long()])
    Zs = sc.transform(Zall)

    def band_idx(split_names, lo, hi):
        out = []
        for s in split_names:
            idx = sp[s]
            out += [i for i in idx if np.isfinite(raw[i]) and lo <= raw[i] < hi]
        return np.array(out, dtype=int)

    ho, tr, mids, labels = [], [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        h = band_idx(["valid", "test"], lo, hi); t = band_idx(["train"], lo, hi)
        ho.append(h); tr.append(t)
        mids.append(float(np.median(raw[h])))
        labels.append(f"{lo}–{hi}")
    print("bands (measured melting point, C):")
    for l, h, t, m in zip(labels, ho, tr, mids):
        print(f"  {l:>10}  held-out {len(h):3d}   train {len(t):4d}   median {m:7.1f}")

    Zho = [Zs[torch.from_numpy(h).long()].numpy() for h in ho]
    Ztr = [Zs[torch.from_numpy(t).long()].numpy() for t in tr]

    # ---- control: real train bands vs real held-out bands
    ctrl = np.array([[sliced_w2(Ztr[i], Zho[j]) for j in range(len(ho))] for i in range(len(tr))])
    ok_ctrl = show(ctrl, labels, labels, "CONTROL — real train band (row) vs real held-out band (column)",
                   "train band")

    out = {"edges": EDGES, "labels": labels, "medians": mids,
           "n_heldout": [int(len(h)) for h in ho], "n_train": [int(len(t)) for t in tr],
           "control": ctrl.tolist(), "control_diag": ok_ctrl, "generated": {}}

    fm = load_cfg_fm(a.tag, a.latent_dim, dev)

    for w in [float(x) for x in a.weights.split(",")]:
        g = CFGVelocity(fm, w=w); G = []
        for T in mids:
            yv = torch.tensor([[(T - mu) / sd, 1.0]]).repeat(a.n, 1)
            with torch.no_grad():
                G.append(sample(g, n=a.n, shape=(a.latent_dim,), y=yv,
                                n_steps=a.ode_steps, method="heun", device=dev).numpy())
        M = np.array([[sliced_w2(G[i], Zho[j]) for j in range(len(ho))] for i in range(len(G))])
        okg = show(M, [f"req {m:.0f}" for m in mids], labels,
                   f"GENERATED at w = {w:g} (row = requested) vs real held-out band (column)",
                   "requested")
        out["generated"][str(w)] = {"matrix": M.tolist(), "diag": okg}

    write_json(a.out, out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
