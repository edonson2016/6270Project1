# fm-chem — minimal flow matching in PyTorch

Standard linear interpolant, no guidance, plain ODE vector field. Built to be
read, not to win benchmarks.

## The whole method

```
train:   x0 ~ source,  x1 ~ data,  t ~ U(0,1)
         x_t = (1-t)*x0 + t*x1
         u_t = x1 - x0
         loss = || v_theta(x_t, t, y) - u_t ||^2

sample:  solve dx/dt = v_theta(x, t, y)  from t=0 (source) to t=1 (data)
```

That's `fm/model.py:loss` and `fm/sampling.py:integrate`. Everything else is plumbing.

## Layout

| file | what it holds |
|---|---|
| `fm/paths.py`     | the interpolant: `x_t` and the target velocity `u_t` |
| `fm/model.py`     | `FlowMatching` — the loss |
| `fm/nets.py`      | `VelocityMLP` — residual MLP, sinusoidal time embedding, FiLM conditioning |
| `fm/sampling.py`  | Euler / Heun / RK4 fixed-step ODE integration |
| `fm/data.py`      | `Standardizer`, `VectorDataset`, `PairedVectorDataset` |
| `fm/train.py`     | `Trainer` — AdamW, warmup+cosine, grad clip, EMA, CSV logging |
| `fm/ema.py`       | weight EMA (sample from these, not the raw weights) |
| `fm/featurize.py` | 3-D geometry → fixed-length vectors, plus geometry diagnostics |
| `fm/latent.py`    | `SeqVAE` — a decodable, regularized latent space to run FM in |

## Step 0 — verify the implementation

```bash
.venv/bin/python scripts/sanity_check.py     # ~8 min on CPU
```

Three quantitative tests on 2-D toys where the answer is known: unconditional
density matching, conditional generation, and data-to-data transport. Run this
after any change to `fm/`.

## Track A — molecular conformers

rMD17: 100k DFT conformations of a fixed-composition molecule, with energies and
forces. A thermalized conformational ensemble — the same problem shape as an
ion-pair ensemble, but a 65 MB download you can check your work against.

```bash
.venv/bin/python scripts/prep_rmd17.py --molecule ethanol --featurizer aligned
.venv/bin/python scripts/train_vectors.py --x1 data/ethanol_aligned_x.npy --out runs/ethanol
.venv/bin/python scripts/eval_conformers.py \
    --samples runs/ethanol/samples.npy --meta data/ethanol_aligned_meta.pkl
```

The `--featurizer` flag is the experiment worth running twice. `aligned` fixes a
frame by Kabsch alignment (3N dims, exactly invertible). `distance` uses pairwise
distances (rotation-invariant by construction, but reconstruction is approximate
and mirror images are indistinguishable). Neither is equivariant.

### The featurization comparison, run

Ethanol, 8k steps, width 256, identical settings. Both featurizations produce
chemically clean conformers, and the distance one is clearly better at the metric
that matters:

| | aligned (D=27) | distance (D=36) | real data |
|---|---|---|---|
| bonds intact | 0.9986 | 0.9986 | — |
| no steric clash | 1.0000 | 1.0000 | — |
| bond mean abs err | 0.0009 Å | 0.0008 Å | — |
| **bond std ratio** | 1.087 | **1.021** | 1.000 |
| variance in top 3 eigenvalues | n/a, exact by construction | 0.9950 | 1.0000 |
| **samples with negative eigenvalues** | n/a | **1.0000** | 0.0010 |
| worst negative eigenvalue | n/a | −0.2229 | −0.000001 |

Working in an invariant space means the model never spends capacity on
orientation, and it shows: over-dispersion of the thermal ensemble drops from 9%
to 2%.

The price is in the bottom three rows. **Every single generated sample has
negative Gram eigenvalues** — each one describes a set of interatomic distances
that no arrangement of atoms in three dimensions can realize. Real data sits at
0.001 (pure numerical noise). MDS repairs each sample silently by projecting onto
the top 3 eigenvalues, so the conformer table looks clean while 0.5% of the
model's actual output is geometric nonsense being quietly discarded.

That is the tradeoff in full. `aligned` is valid 3-D by construction but wastes
capacity learning a frame. `distance` spends nothing on the frame but generates
impossible objects and relies on a repair step to hide it. An equivariant
architecture is what buys you both at once: no wasted capacity *and* validity by
construction. You can now see exactly what you are paying for.

Note also that the two runs' FM losses (0.845 vs 1.196) are **not comparable** —
different dimensionality and different post-standardization geometry give
different irreducible coupling variance. Only the domain metrics compare.

## Track B — flow matching in a learned latent space

```bash
.venv/bin/python scripts/prep_moses.py --n 200000
.venv/bin/python scripts/train_latent_ae.py --epochs 15 --beta 3e-3
.venv/bin/python scripts/train_latent_fm.py --steps 30000
```

The flow matching code is *identical* to Track A — same `VelocityMLP`, same path,
same trainer. Only the data changed. That's the point.

`train_latent_fm.py` always reports an **autoencoder ceiling** next to the model
numbers: the same metrics computed by decoding *real* latents. Low uniqueness has
two causes that look identical in the metrics table, and only this comparison
separates them. If the ceiling is high and the model is low, flow matching
mode-collapsed. If the ceiling is also low, the decoder collapsed and no amount
of flow matching work will help.

A real run from this repo, with the autoencoder deliberately starved (8k
molecules, 3 epochs) to show the failure:

```
metric                      model   AE ceiling       reads on
validity                   1.0000       1.0000        nothing
uniqueness                 0.0600       0.0600       collapse
novelty                    1.0000       1.0000   memorization

DIAGNOSIS: the AE ceiling is low too, so the DECODER collapsed.
```

500 samples, 30 unique molecules, and **validity and novelty both read a perfect
1.0**. Validity is 1.0 because SELFIES guarantees it. Novelty is 1.0 because
nonsense is reliably absent from the training set. A model can score perfectly on
both while being completely broken — this is why you read uniqueness first.

**The central experiment is the `--beta` sweep** (0, 1e-4, 3e-3, 1e-1). At
`beta=0` you have a plain autoencoder: great reconstruction, and FM samples that
decode to garbage because the latent is full of holes the decoder never learned.
At high beta the latent is smooth but reconstruction collapses. Plot
reconstruction accuracy against sample novelty and the tradeoff is one figure.

## Colab

`notebooks/colab_quickstart.ipynb` runs both tracks on a free T4.

## Three things that will bite you

**Standardize your data.** FM from a standard Gaussian works far better when the
target is on a comparable scale. `train_vectors.py` does it for you. Skipping it
is the most common reason a first run looks broken.

**The validation loss is not a quality metric.** `u_t = x1 - x0` is not a
deterministic function of `(x_t, t)`, so the loss floors at a nonzero value set
by the irreducible variance of the coupling — around 0.86 for ethanol, and that
is *convergence*, not failure. Use it to spot divergence and overfitting. Judge
quality with a domain metric on actual samples: bond-length standard deviations
in Track A, uniqueness and novelty in Track B.

**Match the right statistic.** Bond *means* are easy to match and say almost
nothing. Bond *standard deviations* are the real test — they say whether the
model captured the width of the thermal ensemble or collapsed onto the average
structure. Likewise in Track B: validity is ~100% by SELFIES construction and
proves nothing; uniqueness is what catches mode collapse.

## What this does not give you

The ODE time `t` is the path time of the generative process. It is **not**
physical time. The intermediates `x_t` are points on a straight line between a
source sample and a target sample — not reaction coordinates, not transition
states, not folding intermediates, however suggestive they look animated. If you
want physical intermediates you need a model whose time axis is physical (MD, a
Markov state model, or a bridge trained on real trajectory snapshots).

## Data sources

- rMD17 — [figshare 12672038](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038) (per-molecule `.npz`, 65–175 MB each)
- MD22 (larger systems, 400–500 K) — [sgdml.org](http://www.sgdml.org/#datasets)
- MOSES — [molecularsets/moses](https://github.com/molecularsets/moses)
