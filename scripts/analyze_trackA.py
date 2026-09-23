"""Figures and diagnostics for the Track A ethanol runs."""
from __future__ import annotations
import csv, pickle, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
SURF = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": INK3, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 9, "axes.titlesize": 11, "axes.spines.top": False,
    "axes.spines.right": False, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
})


def read_log(p):
    steps, tr, vs, vl = [], [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            steps.append(int(r["step"])); tr.append(float(r["train_loss"]))
            v = float(r["val_loss"])
            if not np.isnan(v):
                vs.append(int(r["step"])); vl.append(v)
    return np.array(steps), np.array(tr), np.array(vs), np.array(vl)


def dihedral(c, i, j, k, l):
    """Signed dihedral angle i-j-k-l in degrees. c: (M, N, 3)."""
    b0 = c[:, i] - c[:, j]
    b1 = c[:, k] - c[:, j]
    b2 = c[:, l] - c[:, k]
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - (b0 * b1).sum(1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(1, keepdims=True) * b1
    x = (v * w).sum(1)
    y = (np.cross(b1, v) * w).sum(1)
    return np.degrees(np.arctan2(y, x))


def load_gen(run, meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta["featurizer"].inverse(np.load(run)), meta


# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(figsize=(7.2, 4.2))
runs = [("runs/ethanol/log.csv", "aligned Cartesian (D=27)", BLUE),
        ("runs/ethanol_distance/log.csv", "pairwise distance (D=36)", ORANGE)]
for path, label, col in runs:
    s, tr, vs, vl = read_log(path)
    ax.plot(s, tr, color=col, lw=2, label=label, solid_capstyle="round")
    ax.plot(vs, vl, "o", color=col, ms=5, mec=SURF, mew=1.5, zorder=3)
    ax.annotate(f"{label}\nfloor {tr[-1]:.3f}", xy=(s[-1], tr[-1]),
                xytext=(-6, 26), textcoords="offset points", ha="right",
                fontsize=8.5, color=INK2, linespacing=1.4)

ax.set_xlabel("training step"); ax.set_ylabel("flow matching loss  ‖v$_θ$ − u$_t$‖²")
ax.set_title("Track A loss curves — ethanol, 8k steps, width 256, depth 4", color=INK, pad=12)
ax.grid(axis="y"); ax.set_axisbelow(True); ax.set_ylim(0.7, 1.9)
ax.legend(frameon=False, loc="upper right", fontsize=8.5, labelcolor=INK2)
ax.text(0.5, -0.20, "Dots are validation loss. The plateaus are CONVERGENCE, not failure: u$_t$ = x$_1$−x$_0$ is not a\n"
        "function of (x$_t$, t), so the loss floors at the coupling's irreducible variance. The two floors differ\n"
        "because of dimensionality and scaling — they are NOT comparable to each other.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color=INK3, linespacing=1.6)
fig.tight_layout(); fig.savefig("runs/fig_loss_curves.png", dpi=170, bbox_inches="tight")
print("wrote runs/fig_loss_curves.png")

# ---------------------------------------------------------------- figure 2
raw = np.load("data/rmd17_ethanol.npz")
real = raw["coords"]
gen_a, _ = load_gen("runs/ethanol/samples.npy", "data/ethanol_aligned_meta.pkl")
gen_d, _ = load_gen("runs/ethanol_distance/samples.npy", "data/ethanol_distance_meta.pkl")

# atoms: 0=C(carbinol) 1=C(methyl) 2=O 8=H(hydroxyl)
series = [("rMD17 reference", real, AQUA, 2.5),
          ("aligned Cartesian", gen_a, BLUE, 2.0),
          ("pairwise distance", gen_d, ORANGE, 2.0)]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
bins = np.linspace(-180, 180, 73)
ctr = 0.5 * (bins[:-1] + bins[1:])

for name, c, col, lw in series:
    h, _ = np.histogram(np.abs(dihedral(c, 8, 2, 0, 1)), bins=np.linspace(0, 180, 61), density=True)
    axes[0].plot(np.linspace(1.5, 178.5, 60), h, color=col, lw=lw, label=name)
    h2, _ = np.histogram(dihedral(c, 2, 0, 1, 5), bins=bins, density=True)
    axes[1].plot(ctr, h2, color=col, lw=lw, label=name)

axes[0].set_xlabel("|H–O–C–C| dihedral  (degrees)"); axes[0].set_xlim(0, 180)
axes[0].set_title("hydroxyl torsion — the conformational degree of freedom", fontsize=9.5, color=INK)
axes[0].text(0.33, 0.93, "gauche\n(~65°)", transform=axes[0].transAxes,
             ha="center", va="top", fontsize=8.5, color=INK3, linespacing=1.4)
axes[0].text(0.955, 0.42, "anti\n(180°)", transform=axes[0].transAxes,
             ha="center", va="top", fontsize=8.5, color=INK3, linespacing=1.4)
axes[1].set_xlabel("O–C–C–H dihedral  (degrees)"); axes[1].set_xlim(-180, 180)
axes[1].set_title("methyl rotation — nearly free, all angles populated", fontsize=9.5, color=INK)

for a in axes:
    a.set_ylabel("probability density"); a.grid(axis="y"); a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
fig.suptitle("What the model generates: the high-temperature thermal ensemble of gas-phase ethanol",
             color=INK, fontsize=11, y=1.02)
fig.tight_layout(); fig.savefig("runs/fig_ethanol_states.png", dpi=170, bbox_inches="tight")
print("wrote runs/fig_ethanol_states.png")

# ---------------------------------------------------------------- diagnostics
print("\n=== what ensemble is this? ===")
E = raw["energies"]; N = real.shape[1]; ndof = 3 * N - 6
kT = E.std() / np.sqrt(ndof / 2)
print(f"energy std {E.std():.2f} kcal/mol over {ndof} vibrational DOF")
print(f"  -> equipartition estimate kT = {kT:.4f} kcal/mol  =>  T ~ {kT/0.0019872:.0f} K")
print(f"  (harmonic estimate; anharmonicity biases it high. A ~300 K ensemble would")
print(f"   give std ~{np.sqrt(ndof/2)*0.0019872*300:.2f} kcal/mol, far below the observed {E.std():.2f}.)")

print("\n=== mode populations, |H-O-C-C| torsion ===")
for name, c, _, _ in series:
    t = np.abs(dihedral(c, 8, 2, 0, 1))
    print(f"  {name:<22s} anti(>120 deg) {np.mean(t > 120):6.1%}   gauche(<120 deg) {np.mean(t <= 120):6.1%}")

print("\n=== the readme's warning, quantified ===")
# The rMD17 file is SHUFFLED, so autocorrelation measured in file order reads
# ~0 and looks reassuring. It is an artifact. Sort by old_indices to recover
# the true trajectory order before measuring anything.
oi = raw["old_indices"]
order = np.argsort(oi)
gaps = np.diff(np.sort(oi))
print(f"  100k structures subsampled from ~{oi.max()+1:,} MD frames "
      f"(median gap {np.median(gaps):.0f} frames)")
c = real[order]
xf = c.reshape(len(c), -1)
xf = (xf - xf.mean(0)) / xf.std(0)
print("  Cartesian autocorrelation, TRUE trajectory order:")
for k in (1, 10, 100, 1000):
    print(f"    lag {k:>5d}: {float((xf[:-k]*xf[k:]).mean()):+.4f}")
tt = np.abs(dihedral(c, 8, 2, 0, 1))
tt = (tt - tt.mean()) / tt.std()
print("  |H-O-C-C| torsion autocorrelation, TRUE trajectory order:")
for k in (1, 20, 100, 1000):
    print(f"    lag {k:>5d}: {float((tt[:-k]*tt[k:]).mean()):+.4f}")
print("  => consecutive rows are near-duplicates. A random train/val split does")
print("     NOT separate them, so the validation loss cannot detect memorization")
print("     here. Ensemble statistics stay valid; the val curve does not.")
