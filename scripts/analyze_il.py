"""Figures and summary tables for the ionic-liquid latent flow matching run.

Reads runs/il/results/results.jsonl (one JSON line per config, appended by
scripts/il_finetune_fm.py) and writes figures next to it.

Rows are deduplicated by `tag`, last one wins, so re-running a config supersedes
its earlier result rather than double-plotting it.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm.flops import fmt as _f

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, INK3, SURF = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": INK3, "axes.linewidth": .8, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e4e3df", "grid.linewidth": .6})

OUT = Path("runs/il/results")
ARM_COL = {"fm": BLUE, "prior": INK3, "ceiling": ORANGE}
ARM_LBL = {"fm": "fm  (N(0,I) → ODE)", "prior": "prior  (N(0,I) direct)",
           "ceiling": "ceiling  (encode real)"}


def load(schema=1):
    """Rows of one schema version, deduplicated by tag (later rows win).

    schema 1: scored on the full corpus, no held-out set (the first runs).
    schema 2: the split protocol -- train/valid/test/test_ion plus the OOD set.
    They are kept apart because their arm names and denominators differ, so
    plotting them on one axis would compare two different measurements.
    """
    f = OUT / "results.jsonl"
    if not f.exists():
        print(f"no results at {f}"); return []
    by_tag = {}
    for line in f.read_text().split("\n"):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if r.get("schema", 1) != schema: continue
        by_tag[r["tag"]] = r
    return sorted(by_tag.values(), key=lambda r: (r["decoder"], r["latent_mode"], r["latent_dim"]))


def label(r):
    return f"{r['decoder']}/{r['latent_mode']}\nd={r['latent_dim']}"


# ------------------------------------------------------------------ ARMS
def fig_arms(R):
    """The central comparison: three arms sharing one decoder, per config."""
    metrics = [("validity", "validity  (RDKit-parseable)"),
               ("uniqueness", "uniqueness  (mode-collapse detector)"),
               ("novel_cation_rate", "novel cation rate")]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))
    x = np.arange(len(R)); w = 0.26
    for ax, (key, title) in zip(axes, metrics):
        for i, arm in enumerate(("prior", "fm", "ceiling")):
            v = [r["arms"][arm][key] for r in R]
            ax.bar(x + (i-1)*w, v, w, color=ARM_COL[arm], edgecolor=SURF, linewidth=1.0,
                   label=ARM_LBL[arm] if key == "validity" else None, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels([label(r) for r in R], fontsize=7.5)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_ylim(0, 1.05); ax.grid(axis="y"); ax.set_axisbelow(True)
    # mark configs whose reconstruction gate failed -- their numbers are suspect
    for ax in axes:
        for i, r in enumerate(R):
            if not r["gate_pass"]:
                ax.axvspan(i-.45, i+.45, color=YELLOW, alpha=.10, zorder=0)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.suptitle("Three arms, one decoder — only the source of z differs "
                 "(shaded = Stage-2 gate failed, read with suspicion)",
                 fontsize=11, color=INK, y=1.04)
    fig.tight_layout(); fig.savefig(OUT/"fig_il_arms.png", dpi=170, bbox_inches="tight")
    print(f"  wrote {OUT/'fig_il_arms.png'}")


# ------------------------------------------------------------- CEILING GAP
def fig_ceiling(R):
    """How much of the reachable validity the velocity field actually captured.

    prior is the floor (right marginals, no structure), ceiling is the decoder's
    own limit. The bar is what FM closed of that gap.
    """
    fig, ax = plt.subplots(figsize=(7.6, 0.62*len(R) + 1.8))
    y = np.arange(len(R))
    for i, r in enumerate(R):
        p, f, c = (r["arms"][a]["validity"] for a in ("prior", "fm", "ceiling"))
        ax.plot([p, c], [i, i], color=INK3, lw=1.2, zorder=1)
        ax.barh(i, max(f-p, 0), left=p, height=.55, color=BLUE, edgecolor=SURF,
                linewidth=1.0, zorder=2)
        ax.scatter([p], [i], s=46, marker="|", color=INK2, zorder=3)
        ax.scatter([c], [i], s=54, marker="D", color=ORANGE, edgecolor=SURF,
                   linewidth=1.0, zorder=3)
        frac = (f-p)/(c-p) if c > p else float("nan")
        ax.text(max(c, f)+.02, i, f"{frac*100:.0f}% of gap" if np.isfinite(frac) else "n/a",
                va="center", fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([label(r).replace("\n", "  ") for r in R], fontsize=8)
    ax.set_xlim(0, 1.18); ax.set_xlabel("validity")
    ax.invert_yaxis(); ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.scatter([], [], s=46, marker="|", color=INK2, label="prior (floor)")
    ax.scatter([], [], s=54, marker="D", color=ORANGE, label="ceiling (decoder limit)")
    ax.bar([], [], color=BLUE, label="what the velocity field added")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.suptitle("The velocity field is not the bottleneck — the decoder is",
                 fontsize=11, color=INK, y=1.0)
    fig.tight_layout(); fig.savefig(OUT/"fig_il_ceiling.png", dpi=170, bbox_inches="tight")
    print(f"  wrote {OUT/'fig_il_ceiling.png'}")


# ---------------------------------------------------------------- d SWEEP
def fig_dsweep(R):
    """Only drawn when some (decoder, latent_mode) was run at 2+ latent dims."""
    groups = {}
    for r in R:
        groups.setdefault((r["decoder"], r["latent_mode"]), []).append(r)
    groups = {k: sorted(v, key=lambda r: r["latent_dim"])
              for k, v in groups.items() if len(v) >= 2}
    if not groups:
        print("  (no latent-dim sweep in results -- skipping fig_il_dsweep)"); return

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    for (dec, mode), rows in groups.items():
        col = cols.pop(0) if cols else INK3
        d = [r["latent_dim"] for r in rows]
        axes[0].plot(d, [r["ft_val_token_acc"] for r in rows], "o-", color=col, lw=2,
                     ms=6, mec=SURF, mew=1.4, label=f"{dec}/{mode}")
        axes[1].plot(d, [r["arms"]["fm"]["validity"] for r in rows], "o-", color=col,
                     lw=2, ms=6, mec=SURF, mew=1.4, label=f"{dec}/{mode} — fm")
        axes[1].plot(d, [r["arms"]["ceiling"]["validity"] for r in rows], "o--", color=col,
                     lw=1.5, ms=4, alpha=.75, mec=SURF, mew=1, label=f"{dec}/{mode} — ceiling")
        axes[2].plot(d, [r["arms"]["fm"]["uniqueness"] for r in rows], "o-", color=col,
                     lw=2, ms=6, mec=SURF, mew=1.4, label=f"{dec}/{mode} — fm")
        axes[2].plot(d, [r["arms"]["prior"]["uniqueness"] for r in rows], "o--", color=col,
                     lw=1.5, ms=4, alpha=.75, mec=SURF, mew=1, label=f"{dec}/{mode} — prior")
    axes[0].axhline(0.90, color=INK3, lw=1.2, ls=":", zorder=1)
    axes[0].annotate("gate 0.90", xy=(0.99, 0.90), xycoords=("axes fraction", "data"),
                     xytext=(0, 3), textcoords="offset points",
                     fontsize=7.5, color=INK2, ha="right", va="bottom")
    axes[0].set_ylabel("Stage-2 val token accuracy")
    axes[0].set_title("does a bigger latent reconstruct better?", fontsize=9.5, color=INK)
    axes[1].set_ylabel("validity")
    axes[1].set_title("solid = flow matching, dashed = decoder ceiling", fontsize=9.5, color=INK)
    axes[2].set_ylabel("uniqueness")
    axes[2].set_title("solid = flow matching, dashed = prior only", fontsize=9.5, color=INK)
    for a in axes:
        a.set_xscale("log", base=2); a.set_xlabel("latent dimension d")
        a.grid(axis="y"); a.set_axisbelow(True)
        a.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    fig.suptitle("Ionic liquids: latent dimension sweep", fontsize=11, color=INK, y=1.03)
    fig.tight_layout(); fig.savefig(OUT/"fig_il_dsweep.png", dpi=170, bbox_inches="tight")
    print(f"  wrote {OUT/'fig_il_dsweep.png'}")


# ----------------------------------------------------------------- TABLES
def tables(R):
    print("\n" + "="*100)
    print("IONIC LIQUIDS — latent flow matching, three arms per config")
    print("="*100)
    print(f"\n  {'tag':<20s}{'dec':>5s}{'mode':>11s}{'d':>5s}{'tok_acc':>9s}"
          f"{'exact':>8s}{'gate':>7s}{'w2':>8s}")
    for r in R:
        print(f"  {r['tag']:<20s}{r['decoder']:>5s}{r['latent_mode']:>11s}{r['latent_dim']:>5d}"
              f"{r['ft_val_token_acc']:>9.4f}{r['ft_val_exact']:>8.4f}"
              f"{'PASS' if r['gate_pass'] else 'FAIL':>7s}{r['w2_fm_vs_z']:>8.3f}")
    print("\n  w2 is sliced Wasserstein between generated and real z, in UNSTANDARDIZED")
    print("  units. The z clouds have different scales per config, so it compares only")
    print("  within a row's own arms -- never across rows.")

    print(f"\n  {'tag':<20s}{'arm':<9s}{'valid':>8s}{'2frag':>8s}{'charge':>8s}"
          f"{'uniq':>8s}{'nov_cat':>9s}{'nov_an':>8s}{'n_uniq':>8s}")
    for r in R:
        for arm in ("fm", "prior", "ceiling"):
            m = r["arms"][arm]
            print(f"  {r['tag'] if arm=='fm' else '':<20s}{arm:<9s}{m['validity']:>8.3f}"
                  f"{m['two_fragment']:>8.3f}{m['charge_balanced']:>8.3f}{m['uniqueness']:>8.3f}"
                  f"{m['novel_cation_rate']:>9.3f}{m['novel_anion_rate']:>8.3f}{m['n_unique']:>8d}")

    print("\n  what the velocity field buys, and how much of the reachable range it took:")
    for r in R:
        p, f, c = (r["arms"][a]["validity"] for a in ("prior", "fm", "ceiling"))
        frac = f"{(f-p)/(c-p)*100:5.1f}% of gap" if c > p else "   n/a (no gap)"
        flag = "" if r["gate_pass"] else "   [gate FAIL]"
        print(f"    {r['tag']:<20s} validity {p:.3f} -> {f:.3f}  (ceiling {c:.3f})  {frac}{flag}")



# ------------------------------------------------------- GENERALIZATION (v2)
SPLIT_COL = {"train": INK3, "valid": AQUA, "test": BLUE, "test_ion": ORANGE, "ood": YELLOW}
SPLIT_LBL = {"train": "train", "valid": "valid", "test": "test (known ions)",
             "test_ion": "test_ion (unseen cations)", "ood": "ood (ILThermo)"}


def fig_generalization(R):
    """What the held-out splits buy: the flow's fit to data it did not train on.

    Left is the one that matters. Before the split existed, sliced W2 could only
    be computed against training latents, so a flow that memorized its 4,790
    points scored perfectly and the metric could not fail.
    """
    if not R:
        print("  (no schema-2 rows -- skipping generalization figures)"); return
    d = [r["latent_dim"] for r in R]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))

    for s in ("train", "valid", "test", "test_ion", "ood"):
        axes[0].plot(d, [r["sw2"][s] for r in R], "o-", color=SPLIT_COL[s], lw=2, ms=6,
                     mec=SURF, mew=1.4, label=SPLIT_LBL[s])
        axes[1].plot(d, [r["ae_recon"][s]["token_acc"] for r in R], "o-", color=SPLIT_COL[s],
                     lw=2, ms=6, mec=SURF, mew=1.4, label=SPLIT_LBL[s])
    axes[0].set_yscale("log"); axes[0].set_ylabel("sliced W₂ (standardized)")
    axes[0].set_title("flow: distance from generated z to each split", fontsize=9.5, color=INK)
    axes[1].set_ylabel("teacher-forced token accuracy")
    axes[1].set_title("autoencoder: reconstruction by split", fontsize=9.5, color=INK)

    for arm, col, lbl in (("prior", INK3, "prior"), ("fm", BLUE, "fm"),
                          ("ceiling_test", ORANGE, "ceiling (test)"),
                          ("ceiling_ood", YELLOW, "ceiling (ood)")):
        axes[2].plot(d, [r["arms"][arm]["plausible"] for r in R], "o-", color=col, lw=2,
                     ms=6, mec=SURF, mew=1.4, label=lbl)
    axes[2].set_ylabel("chemical plausibility")
    axes[2].set_title("generation: plausible ion pairs", fontsize=9.5, color=INK)
    axes[2].axhline(0.99, color=INK3, lw=1.2, ls=":", zorder=1)
    axes[2].annotate("real data ≈ 0.99", xy=(0.99, 0.99), xycoords=("axes fraction", "data"),
                     xytext=(0, -11), textcoords="offset points", fontsize=7.5,
                     color=INK2, ha="right")
    for a in axes:
        a.set_xscale("log", base=2); a.set_xlabel("latent dimension d")
        a.grid(axis="y"); a.set_axisbelow(True)
        a.legend(fontsize=7.5, labelcolor=INK2, frameon=True, facecolor=SURF,
                 edgecolor="none", framealpha=.88)
    fig.suptitle("Held-out evaluation: the flow memorizes more than the autoencoder does",
                 fontsize=11, color=INK, y=1.03)
    fig.tight_layout(); fig.savefig(OUT/"fig_il_generalization.png", dpi=170, bbox_inches="tight")
    print(f"  wrote {OUT/'fig_il_generalization.png'}")


def tables_v2(R):
    if not R: return
    print("\n" + "="*100)
    print("SPLIT PROTOCOL — trained on TRAIN, gated on VALID, scored on TEST / TEST_ION / OOD")
    print("="*100)
    print(f"\n  {'d':>4s} {'gate':>7s} | sliced W2 (standardized), generated z vs each split")
    print(f"  {'':>4s} {'':>7s} | {'train':>7s} {'valid':>7s} {'test':>7s} {'test_ion':>9s}"
          f" {'ood':>7s} {'ion/train':>10s}")
    for r in R:
        s = r["sw2"]
        print(f"  {r['latent_dim']:>4d} {r['ft_val_token_acc']:7.4f} | {s['train']:7.3f} "
              f"{s['valid']:7.3f} {s['test']:7.3f} {s['test_ion']:9.3f} {s['ood']:7.3f} "
              f"{s['test_ion']/max(s['train'],1e-9):9.2f}x")
    print("\n  A flow that had memorized its training latents would read ~0 on the train")
    print("  column and large everywhere else. The ratio is the overfitting signal, and")
    print("  it is only measurable because the splits exist.")

    print(f"\n  {'d':>4s} | {'fm':>7s} {'prior':>7s} {'ceil_tr':>8s} {'ceil_te':>8s}"
          f" {'ceil_ion':>9s} {'ceil_ood':>9s} {'real':>7s}   (chemical plausibility)")
    for r in R:
        A = r["arms"]
        print(f"  {r['latent_dim']:>4d} | {A['fm']['plausible']:7.3f} {A['prior']['plausible']:7.3f} "
              f"{A['ceiling_train']['plausible']:8.3f} {A['ceiling_test']['plausible']:8.3f} "
              f"{A['ceiling_test_ion']['plausible']:9.3f} {A['ceiling_ood']['plausible']:9.3f} "
              f"{A['real_test']['plausible']:7.3f}")

    print(f"\n  {'d':>4s} | recovery of ions never seen in training")
    print(f"  {'':>4s} | {'test_ion cat':>13s} {'test_ion an':>12s} {'ood cat':>8s} {'ood an':>8s}")
    for r in R:
        v, o = r["recovery"]["test_ion"], r["recovery"]["ood"]
        print(f"  {r['latent_dim']:>4d} | {v['cation_recall']:13.3f} {v['anion_recall']:12.3f} "
              f"{o['cation_recall']:8.3f} {o['anion_recall']:8.3f}")
    print("\n  test_ion cations are unseen BY CONSTRUCTION, so that column is the honest")
    print("  'can it invent held-out chemistry' number. The ood columns are not: most ood")
    print("  ions also occur in training, so they mostly measure recall of common ions.")



# ------------------------------------------------------- FM vs DDPM (v2)
GEN_STYLE = {("fm", "heun"):       (BLUE,   "o-",  "flow matching (heun-50, 100 NFE)"),
             ("ddpm", "ancestral"):(ORANGE, "s-",  "DDPM ancestral (1000 NFE)"),
             ("ddpm", "ddim"):     (AQUA,   "^--", "DDPM DDIM-50 (50 NFE)")}


def fig_fm_vs_ddpm(R):
    """Same latent cloud, same splits, same network -- only Stage 4 differs."""
    G = [r for r in R if r.get("generative") and r["tag"].startswith("gen_")]
    if not G:
        print("  (no fm-vs-ddpm rows -- skipping)"); return
    groups = {}
    for r in G:
        groups.setdefault((r["generative"], r["sampler"]), []).append(r)
    groups = {k: sorted(v, key=lambda r: r["latent_dim"]) for k, v in groups.items()}

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))
    for k, rows in groups.items():
        if k not in GEN_STYLE: continue
        col, mk, lbl = GEN_STYLE[k]
        d = [r["latent_dim"] for r in rows]
        axes[0].plot(d, [r["arms"]["fm"]["plausible"] for r in rows], mk, color=col, lw=2,
                     ms=6, mec=SURF, mew=1.4, label=lbl)
        axes[2].plot(d, [r["sw2"]["test_ion"] for r in rows], mk, color=col, lw=2,
                     ms=6, mec=SURF, mew=1.4, label=lbl)
        axes[1].scatter([r["cost"]["flops_sample_per"] for r in rows],
                        [r["arms"]["fm"]["plausible"] for r in rows],
                        s=[18 + 1.1 * r["latent_dim"] for r in rows], color=col,
                        edgecolor=SURF, linewidth=1.2, label=lbl, zorder=3)
    # the decoder ceiling is shared by construction, so it is one line for all models
    any_rows = next(iter(groups.values()))
    axes[0].plot([r["latent_dim"] for r in any_rows],
                 [r["arms"]["ceiling_test"]["plausible"] for r in any_rows],
                 ":", color=INK3, lw=1.6, label="decoder ceiling (shared)")

    axes[0].set_xscale("log", base=2); axes[0].set_xlabel("latent dimension d")
    axes[0].set_ylabel("chemical plausibility")
    axes[0].set_title("quality: flow matching wins at every d", fontsize=9.5, color=INK)
    axes[1].set_xscale("log"); axes[1].set_xlabel("FLOPs to generate one sample")
    axes[1].set_ylabel("chemical plausibility")
    axes[1].set_title("cost vs quality (marker size = d)", fontsize=9.5, color=INK)
    axes[2].set_xscale("log", base=2); axes[2].set_xlabel("latent dimension d")
    axes[2].set_ylabel("sliced W₂ vs test_ion")
    axes[2].set_title("latent fit: nearly tied — and so not predictive",
                      fontsize=9.5, color=INK)
    for a in axes:
        a.grid(axis="y"); a.set_axisbelow(True)
        a.legend(fontsize=7.5, labelcolor=INK2, frameon=True, facecolor=SURF,
                 edgecolor="none", framealpha=.88)
    fig.suptitle("Flow matching vs DDPM on the same latent cloud "
                 "(identical network, optimizer, steps and decoder)",
                 fontsize=11, color=INK, y=1.03)
    fig.tight_layout(); fig.savefig(OUT/"fig_il_fm_vs_ddpm.png", dpi=170, bbox_inches="tight")
    print(f"  wrote {OUT/'fig_il_fm_vs_ddpm.png'}")


def tables_gen(R):
    G = [r for r in R if r.get("generative") and r["tag"].startswith("gen_")]
    if not G: return
    G = sorted(G, key=lambda r: (r["latent_dim"], r["generative"], r["sampler"]))
    print("\n" + "="*100)
    print("FLOW MATCHING vs DDPM — same z cloud, same splits, same net, same 8000 steps")
    print("="*100)
    print(f"\n  {'d':>4s} {'model':>5s} {'sampler':>10s} | {'plaus':>6s} {'uniq':>6s} "
          f"{'sw2_ion':>8s} {'sw2_ood':>8s} | {'NFE':>5s} {'sampFLOP':>9s} "
          f"{'train_s':>8s} {'samp_s':>7s}")
    for r in G:
        c, s = r["cost"], r["sw2"]
        print(f"  {r['latent_dim']:>4d} {r['generative']:>5s} {r['sampler']:>10s} | "
              f"{r['arms']['fm']['plausible']:6.3f} {r['arms']['fm']['uniqueness']:6.3f} "
              f"{s['test_ion']:8.3f} {s['ood']:8.3f} | {c['nfe_per_sample']:5d} "
              f"{_f(c['flops_sample_per']):>9s} {c['train_secs']:8.0f} {c['sample_secs']:7.1f}")
    print("\n  plausibility gap, flow matching minus DDPM ancestral:")
    for d in sorted({r["latent_dim"] for r in G}):
        f = next((r for r in G if r["latent_dim"] == d and r["generative"] == "fm"), None)
        p = next((r for r in G if r["latent_dim"] == d and r["sampler"] == "ancestral"), None)
        if f and p:
            a, b = f["arms"]["fm"]["plausible"], p["arms"]["fm"]["plausible"]
            print(f"    d={d:>3d}  {a:.3f} vs {b:.3f}   {a-b:+.3f}  ({(a-b)/b:+.1%} relative)"
                  f"   at {p['cost']['nfe_per_sample']//f['cost']['nfe_per_sample']}x less sampling cost")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    R1, R2 = load(schema=1), load(schema=2)
    if R1:
        fig_arms(R1); fig_ceiling(R1); fig_dsweep(R1)
    fig_generalization(R2); fig_fm_vs_ddpm(R2)
    if R1: tables(R1)
    tables_v2(R2); tables_gen(R2)
