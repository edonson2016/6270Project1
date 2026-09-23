"""Figures and summary tables for the overnight experiments."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, INK3, SURF = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": INK3, "axes.linewidth": .8, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e4e3df", "grid.linewidth": .6})
OUT = Path("runs/overnight")


def load(p):
    f = OUT / p / "results.json"
    return json.loads(f.read_text()) if f.exists() else []


# ------------------------------------------------------------------ EXP 2
def exp2():
    R = load("exp2")
    if not R:
        print("exp2: no results"); return
    mols = sorted({r["molecule"] for r in R}, key=lambda m: [r for r in R if r["molecule"]==m][0]["dof"])
    print("\n" + "="*84)
    print("EXPERIMENT 2 — conformer Pareto: learned latent vs hand-chosen features")
    print("="*84)

    # --- fig: AE reconstruction vs latent dim, with the 3N-6 line
    fig, axes = plt.subplots(1, len(mols), figsize=(4.2*len(mols), 3.8), squeeze=False)
    for ax, mol in zip(axes[0], mols):
        rows = sorted([r for r in R if r["molecule"]==mol and r["arm"]=="latent"],
                      key=lambda r: r["latent_dim"])
        seen, d, rc = set(), [], []
        for r in rows:
            if r["latent_dim"] in seen: continue
            seen.add(r["latent_dim"]); d.append(r["latent_dim"]); rc.append(r["ae_recon"])
        dof = rows[0]["dof"] if rows else 0
        ax.plot(d, rc, "o-", color=BLUE, lw=2, ms=6, mec=SURF, mew=1.5, label="AE recon MSE")
        ax.axvline(dof, color=ORANGE, lw=2, ls="--", label=f"3N−6 = {dof}")
        ax.set_yscale("log"); ax.set_xlabel("latent dimension"); ax.set_ylabel("reconstruction MSE")
        ax.set_title(f"{mol}  (N={rows[0]['n_atoms']} atoms)", fontsize=10, color=INK)
        ax.grid(axis="y"); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Does the autoencoder knee sit at the true degrees of freedom?",
                 fontsize=11, color=INK, y=1.03)
    fig.tight_layout(); fig.savefig(OUT/"fig_exp2_intrinsic_dim.png", dpi=170, bbox_inches="tight")
    print(f"  wrote {OUT/'fig_exp2_intrinsic_dim.png'}")

    # --- fig: Pareto, accuracy vs FLOPs
    fig, axes = plt.subplots(1, len(mols), figsize=(4.4*len(mols), 3.9), squeeze=False)
    for ax, mol in zip(axes[0], mols):
        rows = [r for r in R if r["molecule"]==mol]
        for arm, col, mk in (("handcrafted:aligned", ORANGE, "s"),
                             ("handcrafted:distance", YELLOW, "^"),
                             ("latent", BLUE, "o")):
            pts = [(r["flops_sample"], r["bond_std_ratio"], r) for r in rows if r["arm"]==arm]
            if not pts: continue
            ax.scatter([p[0] for p in pts], [abs(p[1]-1.0) for p in pts], s=54, marker=mk,
                       color=col, edgecolor=SURF, linewidth=1.2,
                       label=arm.replace("handcrafted:", "hand: "), zorder=3)
        pass  # (no zero line: this is a log axis)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("FLOPs to generate one sample"); ax.set_ylabel("|bond std ratio − 1|  (lower better)")
        ax.set_title(f"{mol}", fontsize=10, color=INK)
        ax.grid(True); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Accuracy vs sampling cost — is there a Pareto frontier?",
                 fontsize=11, color=INK, y=1.03)
    fig.tight_layout(); fig.savefig(OUT/"fig_exp2_pareto.png", dpi=170, bbox_inches="tight")
    print(f"  wrote {OUT/'fig_exp2_pareto.png'}")

    for mol in mols:
        rows = sorted([r for r in R if r["molecule"]==mol],
                      key=lambda r: (r["arm"], r.get("latent_dim") or 0, r.get("width", 0)))
        print(f"\n{mol}  (N={rows[0]['n_atoms']}, 3N-6={rows[0]['dof']})")
        print(f"  {'arm':<22s}{'d':>5s}{'W':>5s}{'recon':>10s}{'valid':>8s}{'std_ratio':>11s}"
              f"{'sampleFLOP':>12s}{'trainFLOP':>12s}")
        for r in rows:
            d = r.get("latent_dim"); d = "-" if d is None else str(d)
            print(f"  {r['arm']:<22s}{d:>5s}{r.get('width',0):>5d}{r['ae_recon']:>10.5f}"
                  f"{r['frac_valid']:>8.4f}{r['bond_std_ratio']:>11.4f}"
                  f"{_f(r['flops_sample']):>12s}{_f(r['flops_train_total']):>12s}")


def _f(n):
    for u, d in (("P",1e15),("T",1e12),("G",1e9),("M",1e6),("K",1e3)):
        if abs(n) >= d: return f"{n/d:.2f}{u}"
    return f"{n:.0f}"


# ------------------------------------------------------------------ EXP 1
def exp1():
    R = load("exp1")
    if not R:
        print("exp1: no results"); return
    bins = ["short", "medium", "long"]
    bins = [b for b in bins if any(r["bin"]==b for r in R)]
    print("\n" + "="*92)
    print("EXPERIMENT 1 — latent dimension vs molecule size")
    print("="*92)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    cols = {"short": AQUA, "medium": BLUE, "long": ORANGE}
    for b in bins:
        rows = sorted([r for r in R if r["bin"]==b], key=lambda r: r["latent_dim"])
        d = [r["latent_dim"] for r in rows]
        axes[0].plot(d, [r["token_acc"] for r in rows], "o-", color=cols[b], lw=2, ms=6,
                     mec=SURF, mew=1.4, label=f"{b} (len {rows[0]['mean_len']:.0f})")
        axes[1].plot(d, [r["active_units"]/r["latent_dim"] for r in rows], "o-", color=cols[b],
                     lw=2, ms=6, mec=SURF, mew=1.4, label=b)
        axes[2].plot(d, [r["fm_uniqueness"] for r in rows], "o-", color=cols[b], lw=2, ms=6,
                     mec=SURF, mew=1.4, label=f"{b} — FM")
        axes[2].plot(d, [r["prior_uniqueness"] for r in rows], "o--", color=cols[b], lw=1.5,
                     ms=4, alpha=.75, mec=SURF, mew=1, label=f"{b} — prior only")
    axes[0].set_ylabel("token reconstruction accuracy"); axes[0].set_title(
        "capacity: does a bigger molecule need a bigger latent?", fontsize=9.5, color=INK)
    axes[1].set_ylabel("fraction of latent dims active"); axes[1].set_title(
        "posterior collapse check", fontsize=9.5, color=INK); axes[1].set_ylim(-.05, 1.05)
    axes[2].set_ylabel("uniqueness of generated molecules"); axes[2].set_title(
        "solid = flow matching, dashed = prior sampling only", fontsize=9.5, color=INK)
    for a in axes:
        a.set_xscale("log", base=2); a.set_xlabel("latent dimension")
        a.grid(axis="y"); a.set_axisbelow(True)
        a.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    fig.suptitle("MOSES: latent dimension vs molecular complexity", fontsize=11, color=INK, y=1.03)
    fig.tight_layout(); fig.savefig(OUT/"fig_exp1_scaling.png", dpi=170, bbox_inches="tight")
    print(f"  wrote {OUT/'fig_exp1_scaling.png'}")

    print(f"\n  {'bin':<8s}{'d':>5s}{'tok_acc':>9s}{'exact':>8s}{'active':>8s}"
          f"{'uniq_FM':>9s}{'uniq_pri':>9s}{'uniq_ceil':>10s}{'w2(q,N)':>9s}{'w2(fm,q)':>10s}{'sampFLOP':>11s}")
    for b in bins:
        for r in sorted([x for x in R if x["bin"]==b], key=lambda x: x["latent_dim"]):
            print(f"  {b:<8s}{r['latent_dim']:>5d}{r['token_acc']:>9.4f}{r['exact_recon']:>8.3f}"
                  f"{r['active_units']:>5d}/{r['latent_dim']:<2d}{r['fm_uniqueness']:>9.3f}"
                  f"{r['prior_uniqueness']:>9.3f}{r['ceil_uniqueness']:>10.3f}"
                  f"{r['w2_qz_vs_prior']:>9.3f}{r['w2_fm_vs_qz']:>10.3f}{_f(r['flops_sample']):>11s}")

    gains = [(r["fm_uniqueness"] - r["prior_uniqueness"]) for r in R]
    print(f"\n  flow matching vs prior sampling, uniqueness gain: "
          f"mean {np.mean(gains):+.3f}, min {np.min(gains):+.3f}, max {np.max(gains):+.3f}")
    print("  (this is what the velocity field buys over sampling the VAE prior directly)")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    exp2(); exp1()
