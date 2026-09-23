# Ionic-liquid latent flow matching

Unconditional generation of ionic-liquid cation/anion pairs by flow matching in a
learned latent space. Written for whoever picks this up next — including the
dead ends, because two of them cost real time and are easy to repeat.

---

## 1. The architecture, and what `z` means

```
SMILES ──> [frozen ChemBERTa] ──> mean-pool e (768)
       ──> MLP_down ──> z (d)          <-- flow matching lives HERE
       ──> MLP_up   ──> u
       ──> decoder  ──> SMILES
```

| symbol | what | shape | trained? |
|---|---|---|---|
| `h` | per-token hidden states | (L, 768) | frozen |
| `e` | mean-pooled embedding | (768,) | frozen, no params |
| **`z`** | **the flow matching latent** | **(d,)** | trained (Stages 1–2) |
| `u` | decoder conditioning | (512,) | trained (Stages 1–2) |

`z` is the `d`-dimensional bottleneck and the *only* thing flow matching sees. It
is **not** the 768-dim pooled vector. Inside Stage 4 the dataset is literally an
array of shape `(4790, d)` — the same role a 27-dim ethanol conformer played in
the conformer experiments.

At generation, ChemBERTa / mean-pool / `MLP_down` are **not used at all**:

```
N(0, I_d) ──ODE──> z ──MLP_up──> u ──decoder──> SMILES
```

The encoder's entire job is to have defined a good `z`-cloud during training.

---

## 2. Stage order, and why it is forced

| stage | trains | frozen | objective | script |
|---|---|---|---|---|
| 0 | — | all | embed corpora once, cache | `il_embed.py` |
| 1 | `MLP_down`, `MLP_up`, decoder | ChemBERTa | reconstruct MOSES + IL ions | `il_pretrain.py` |
| 2 | same | ChemBERTa | reconstruct the 4,790 IL pairs | `il_finetune_fm.py` |
| 3 | — | all | encode ILs → fixed `z` cloud | `il_finetune_fm.py` |
| 4 | velocity field only | **everything above** | `‖v_θ(z_t,t) − u_t‖²` | `il_finetune_fm.py` |
| 5 | — | all | generate + score | `il_finetune_fm.py` |

**The decoder never receives a gradient from the flow matching loss.** The stages
are strictly one-directional. Flow matching must learn to sample a latent space
that already exists and is already decodable; train against a moving latent and
the target distribution shifts underneath it.

**Stages 1 and 2 are the same operation on different data.** Same parameters,
same reconstruction loss, same loop — compare the training step in
`il_pretrain.py` with the one in `il_finetune_fm.py` and they are the same line.
"Pretrain" and "fine-tune" name a data curriculum here, not a change of
objective or architecture, and more curriculum phases can be added freely
(§6.9 adds one).

What is genuinely forced is the **latent freeze**. Stage 3 fixes whatever
geometry the autoencoder ends on, and Stage 4 must model that. So the privileged
moment is not "Stage 2" but *the last phase before the freeze* — whatever the
autoencoder sees last decides the distribution flow matching is asked to learn.
That is why §6.9's phase 2b exists: it ends the curriculum on the 4,790 observed
pairs, so the frozen latent reflects real ILs rather than the 103k enumerated
combinations that preceded it.

Consequence for debugging: failures are attributable. A bad decoder cannot be
rescued by flow matching, and a badly organized latent will be faithfully
reproduced by flow matching. Read the three arms (below) to tell which you have.

---

## 3. Why the decoder is pretrained

The decoder must learn two separable things:

1. **SMILES grammar** — matched parens, matched ring digits, legal valences. Generic, data-hungry.
2. **IL chemistry** — which ions exist, charge patterns, the `.` separator.

With only 4,790 IL pairs, a from-scratch decoder spends everything on (1). Prior
experiments here showed ~0.75 token accuracy yields **0.000** exact
reconstruction, and that ~98% token accuracy is needed for 50% exact
reconstruction at length 36.

**It is also the only place pretrained generative knowledge can enter.**
ChemBERTa's knowledge lives in its transformer body and never reaches the
decoder — the decoder is new. Without decoder pretraining, "pretrained" would
describe only the latent geometry.

Pretraining is a **full autoencoder**, not a language model. If you pretrain the
decoder unconditionally and bolt on `z` afterwards, you train exactly the habit
you are trying to prevent: writing plausible molecules without consulting `z`.

The pretraining corpus is MOSES **plus the 16,616 ions in
`il_cations.csv`/`il_anions.csv`**, so charges and bracket atoms appear before
the fine-tune rather than being introduced cold.

Be precise about what those ions are: they are a **DFT virtual-screening
library**, not the inventory of ions in real ILs. Their columns are `homo-fopt`,
`lumo-fopt`; they average 29.2 heavy atoms against 13.6 for observed IL cations;
they include species like `[SbH3-]` and aryl-silane aluminates; and **0 of the
9,172 cations appear in any observed pair**. They serve their stated purpose
(charges, brackets, ionic SMILES grammar) and nothing more. Enumerating training
pairs from them would build the wrong chemical distribution — §6.9 enumerates
from the *observed* ion inventory instead.

---

## 4. Posterior collapse — what applies here and what doesn't

**Classic posterior collapse is a VAE phenomenon.** The loss `recon + β·KL` pays
the model to make `z` carry zero information, because the cheapest way to match
the prior is to ignore the input. An autoregressive decoder is strong enough to
reconstruct without `z`, so the model settles there: KL → 0, latent dead.

**This architecture has no KL term** (the bottleneck is deterministic), so that
mechanism is absent.

**The sibling failure does apply: the teacher-forcing shortcut.** During training
the AR decoder sees true previous tokens and can predict token *t* from tokens
*<t*, leaning on `z` only lightly. At generation there are no true previous
tokens and it collapses. Symptom: good teacher-forced reconstruction, poor
free-running generation.

`word_dropout=0.25` blanks the decoder's own previous tokens, forcing it to
consult `z`. The **NAR decoder has no left-to-right path**, so no shortcut exists
and word dropout is set to 0 for that arm — measuring whether it recovers the
accuracy dropout costs is one of the run's questions.

---

## 5. Arms and metrics

Three arms share one decoder; only the source of `z` differs:

| arm | `z` from | isolates |
|---|---|---|
| `fm` | `N(0,I)` → ODE | the model |
| `prior` | `N(0,I)` directly | what the velocity field adds |
| `ceiling` | `encode(real)` | decoder health, upper bound |

Plus a **random-`z` control** (`--latent-mode random`): each IL gets a fixed
random `z ~ N(0,I)` and the decoder memorizes that arbitrary mapping. Same
pretrained decoder, so only latent *organization* varies.

- If ChemBERTa ≫ random, organization is doing real work.
- If ChemBERTa ≈ random, the encoder contributes nothing and the plan should change.

**Novelty must be split.** Observed pairs are only **0.57%** of the 842,622-cell
cation×anion grid, so ~99.4% of any recombination is trivially "novel."
`novel_combination` is therefore near-saturated and nearly meaningless;
`novel_cation_rate` / `novel_anion_rate` are the real tests.

Uniqueness is a **mode-collapse detector, not a quality score** — an undertrained
decoder making random token errors produces *more* unique molecules, so poor
reconstruction inflates it.

---

## 6. Results — full run (18 Sep 2026)

Two CPU-only sessions (12 cores, no GPU), logged to `runs/il/master.log`:
`run_il_overnight.sh` 02:06→04:27 for the `d=64` configs, then
`run_il_sweep.sh` 12:01→16:04 for the missing 2×2 cell and the `d` sweep.
Seven configs total. Regenerate every figure and table below with:

```bash
.venv/bin/python scripts/analyze_il.py
```

| tag | decoder | latent | d | tok_acc | exact | gate |
|---|---|---|---|---|---|---|
| `ar_chemberta_d16` | AR | ChemBERTa | 16 | 0.9134 | 0.0594 | **PASS** |
| `ar_chemberta_d32` | AR | ChemBERTa | 32 | 0.9181 | 0.0709 | **PASS** |
| `ar_chemberta` | AR | ChemBERTa | 64 | 0.9241 | 0.0915 | **PASS** |
| `ar_chemberta_d128` | AR | ChemBERTa | 128 | **0.9262** | **0.0960** | **PASS** |
| `ar_random` | AR | random | 64 | 0.8239 | 0.0000 | FAIL |
| `nar_chemberta` | NAR | ChemBERTa | 64 | 0.7394 | 0.0544 | FAIL |
| `nar_random` | NAR | random | 64 | 0.1913 | 0.0000 | FAIL |

### 6.1 The velocity field works, and it is not the bottleneck

On every config that passed the gate, flow matching closes most of the distance
between the `prior` floor and the decoder `ceiling`:

| d | prior | **fm** | ceiling | gap closed |
|---|---|---|---|---|
| 16 | 0.413 | **0.744** | 0.787 | **88.4%** |
| 32 | 0.392 | **0.705** | 0.799 | 76.9% |
| 64 | 0.391 | **0.676** | 0.805 | 68.9% |
| 128 | 0.342 | **0.641** | 0.805 | 64.7% |

The size of that gain is the point. The `prior` arm is `N(0,I)` pushed back
through the same `Standardizer`, so it *already* has the right per-dimension
mean and variance. Everything flow matching adds is **correlation structure and
non-Gaussianity** in the z-cloud. What remains above the fm bar is decoder
error, not flow error. See `fig_il_ceiling.png`.

### 6.2 The 2×2 — latent organization is load-bearing, and `ar_random`'s validity was a mirage

| | ChemBERTa `z` | random `z` |
|---|---|---|
| **AR** | valid 0.676, uniq 0.732 | valid **0.987**, uniq **0.029** |
| **NAR** | valid 0.086, uniq 0.820 | valid **0.000**, uniq 0.000 |

`ar_random` looks like the best model in the study on validity alone. It is the
worst. 2,000 samples give 57 unique molecules, 813 of them the single string
`CCCC[n+]1ccn(C)c1.O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F` ([BMIM][NTf2], the
most common IL in the corpus). The decoder could not memorize 4,790 arbitrary
mappings, so it ignored `z` and became an unconditional language model.

**`nar_random` is what proves that reading.** Strip the latent's organization
*and* the left-to-right path and the model produces **zero** parseable molecules
in all three arms, ceiling included, at 0.1913 token accuracy. The two
handicaps compose far worse than either alone — because `ar_random`'s 0.987 was
never latent-driven at all. It was the GRU writing plausible SMILES from its own
previous tokens, which is exactly the habit §3 and §4 are written to prevent.
The NAR decoder has no such fallback, so when `z` carries nothing, nothing comes
out. The floor cell is the control that makes the other three legible.

This also answers §4's open question negatively: removing the teacher-forcing
shortcut did **not** recover what word dropout costs. NAR pretraining plateaued
at 0.672 val accuracy against AR's 0.814, and its z-cloud came out compressed
(per-dim std [0.110, 0.425] vs [1.008, 2.945]). One 64-dim vector broadcast to
`L` positions cannot carry a 36-token molecule at this capacity. The shortcut is
real; the shortcut-free architecture is still worse. **Word dropout on an AR
decoder is the better trade.**

### 6.3 The `d` sweep — the default was on the wrong side of the optimum

`fig_il_dsweep.png`. Reconstruction and generation want opposite things:

| d | tok_acc | fm valid | fm uniq | fm novel_cat | z per-dim std |
|---|---|---|---|---|---|
| 16 | 0.9134 | **0.744** | 0.654 | 0.235 | [2.49, 5.91] |
| 32 | 0.9181 | 0.705 | 0.655 | 0.222 | [1.68, 4.33] |
| 64 | 0.9241 | 0.676 | 0.732 | 0.287 | [1.01, 2.95] |
| 128 | **0.9262** | 0.641 | 0.734 | 0.301 | [0.73, 2.49] |

**Token accuracy rises monotonically with `d`; fm validity falls monotonically
with `d`.** The decoder keeps getting better and the samples keep getting worse,
because the thing that degrades is the *density estimation*, not the decoding.
A 16-dim cloud of 4,790 points is a far easier distribution for the velocity
field to learn than a 128-dim one, and at these sequence lengths the decoder
barely misses the lost capacity — the ceiling only moves 0.787→0.805 across a
factor of 8 in `d`.

So `d=64`, chosen as a default and frozen as a scope cut, sat on the wrong side
of the optimum for generation. **`d=16` is the better operating point**, and the
sweep has not yet found the bottom — `d=8` and `d=4` are the obvious next
points, and they are the cheapest runs in the study.

**But do not read this as "smaller is simply better."** Diversity moves the
other way: uniqueness steps from 0.654 at `d≤32` to 0.734 at `d≥64`, and novel
cation rate from ~0.23 to 0.30. Small `d` buys well-formed molecules drawn from
a narrower region; large `d` buys coverage at the cost of validity. Which end
you want depends on whether the downstream use is screening (take `d=16`) or
exploration (take `d=64`+). That tradeoff, not a single best `d`, is the result.

This also runs *opposite* to the Track-B MOSES sweep, where uniqueness rose with
`d` up to 128 and validity did not collapse. Different decoder and a 45× larger
fine-tune set, so it is not a contradiction — it is a reminder that latent-size
guidance does not transfer across dataset scale.

### 6.4 Three ways to misread these tables

**`w2_fm_vs_z` does not compare across rows — including across the sweep.** It
is computed on *unstandardized* `z`, and the clouds have systematically
different scales (right-hand column above: the per-dim std shrinks as `d` grows).
The sweep's tidy-looking 0.484 → 0.190 trend is mostly that scale change, not
improving fit. Only within-row arm comparisons mean anything. (Same function,
minor: it references `Z[:n]` in dataset order while the `ceiling` arm uses a
random permutation. The ordering is not badly stratified — 262 vs 276 distinct
anions across the split — so this is cosmetic, but a `randperm` there would be
free.)

**Uniqueness is not a quality score.** At `d=64` the AR ceiling's uniqueness
(0.672) sits *below* the fm arm's (0.732). That is not flow matching beating its
own ceiling; it is the inflation §5 warns about, where reconstruction errors
manufacture unique strings. Always read it next to `validity`.

**A gate-FAIL row's arm metrics are not comparable to a gate-PASS row's.** Three
of the seven configs failed the Stage-2 reconstruction gate. Their numbers are
recorded because the *pattern* across arms is informative — that is the whole
argument of §6.2 — but no individual number from them belongs in a comparison
against the ChemBERTa/AR rows. `fig_il_arms.png` shades them for this reason.

### 6.5 Held-out evaluation — what the splits changed

§6.1–6.3 scored every model on the corpus it trained on. There was no held-out
set anywhere: Stage 3 encoded all 4,790 ILs and Stage 4 trained the flow on all
of them, so `w2_fm_vs_z` compared generated latents against *training* latents
and **could not detect memorization even in principle**. The split protocol
(`fm/il_eval.make_splits`) fixes that. Re-run of AR/ChemBERTa at all four `d`,
18 Sep 19:36→20:54:

| split | n | role |
|---|---|---|
| `train` | 3,447 | fits the autoencoder **and** the flow |
| `valid` | 431 | the only set touching the gate or any monitoring |
| `test` | 431 | held-out pairs, ions mostly seen — recombination |
| `test_ion` | 481 | **241 cations, zero overlap with train** — generalization |
| `ood` | 227 | ILThermo/PubChem, a different curation pipeline |

`test_ion` is carved by whole cation *before* the random split, with an assert on
leakage. That axis works because of an asymmetry in the corpus: **69% of cations
are singletons** (1,544 of 2,253), so holding cations out is cheap, whereas the
commonest single anion covers 913 pairs and holding anions out is not.

#### The flow overfits its latent cloud; the autoencoder barely does

Sliced W₂, standardized so the columns are comparable (`fig_il_generalization.png`):

| d | train | valid | test | test_ion | ood | test_ion / train |
|---|---|---|---|---|---|---|
| 16 | **0.047** | 0.120 | 0.087 | 0.144 | **0.489** | 3.08× |
| 32 | 0.051 | 0.116 | 0.109 | 0.139 | 0.505 | 2.74× |
| 64 | 0.057 | 0.120 | 0.092 | 0.143 | 0.465 | 2.49× |
| 128 | 0.058 | 0.109 | 0.116 | 0.131 | 0.506 | 2.28× |

**The flow sits 2.3–3.1× closer to the latents it trained on than to held-out
ones.** That is the number the old protocol structurally could not produce, and
it is the honest read on Stage 4: the velocity field has partly memorized a
3,447-point cloud rather than learned the distribution behind it. Nothing in
§6.1–6.3 is wrong, but every `w2` there was measured against training data and
should be read as a fit statistic, not a generalization one.

The autoencoder is in much better shape. Teacher-forced reconstruction falls
only 0.924 → 0.886 from train to `test_ion` at d=16 — a 0.038 gap against the
flow's 3×. **The two halves of this architecture generalize very differently,
and only per-split measurement separates them.**

Note the overfitting ratio *shrinks* as `d` grows (3.08× → 2.28×). That is not
the flow generalizing better at large `d`; it is the flow fitting everything
worse, with train W₂ rising 0.047 → 0.058 while the held-out columns stay flat.

#### The `d` trend survives, on held-out data

Chemical plausibility of generated pairs — stricter than the old `validity`,
since it also requires net charge zero, a learned element whitelist and no
radicals:

| d | fm | prior | ceiling_test | ceiling_ood | real (calibration) |
|---|---|---|---|---|---|
| 16 | **0.651** | 0.304 | 0.689 | 0.855 | 0.991 |
| 32 | 0.636 | 0.282 | 0.682 | 0.819 | 0.991 |
| 64 | 0.620 | 0.283 | 0.710 | 0.824 | 0.991 |
| 128 | 0.572 | 0.249 | 0.668 | 0.736 | 0.991 |

Monotone in `d`, same direction as §6.3, now measured against held-out ceilings.
**d=16 remains the better operating point for generation.**

#### The OOD set is chemically easier and geometrically farther — at the same time

This is the most interesting dissociation in the run, and it is a warning about
how "out of distribution" gets used:

- In **latent** space the OOD set is far: W₂ ≈ 0.49, roughly **10× the train
  column** and 3.5× `test_ion`. The flow puts almost no mass there.
- In **chemistry** it is *easier* than the internal held-out sets. `ceiling_ood`
  plausibility (0.736–0.855) beats `ceiling_test` (0.668–0.710), and OOD
  reconstruction (0.904–0.913) beats `test_ion` (0.886–0.901).

The cause is composition: the OOD set is **77% imidazolium** (174 of 227), the
single commonest family in training, whereas `test_ion` is deliberately built
from cations the model has never seen. So ILThermo is out of distribution in
*provenance and latent geometry* but in-distribution in *chemistry*. A different
source is not automatically a harder test, and the two tiers must be read apart:
183 of the 227 carry a genuinely unseen ion, and that is the subset that tests
chemistry.

Checked and ruled out as an artifact: the OOD distance is not a SMILES
formatting effect. Both corpora are RDKit-canonical at essentially the same rate
(train 89.0%, ood 87.7%), so the embeddings are not being shifted by
representation. It is a real distributional difference — molecular size, most
visibly, with heavy-atom W₁ of 3.53 against test's 0.47.

#### Can it invent chemistry it never saw? Mostly not

`recovery` — the fraction of held-out ions that unconditional sampling ever
reaches:

| d | test_ion cation | test_ion anion |
|---|---|---|
| 16 | 0.041 | 0.385 |
| 64 | 0.058 | 0.413 |
| 128 | 0.058 | 0.404 |

`test_ion` cations are unseen by construction, so that column is the honest
number: **the model recovers about 5% of them.** The anion column is high only
because anions are heavily reused and mostly *were* in training. The OOD
recovery columns in the table are weaker evidence still, since most OOD ions
also occur in training — they largely measure recall of common ions.

#### One bug this found in the old scorer

`score()` checked `max(q) > 0 and min(q) < 0`, i.e. that one fragment is a cation
and one an anion. It never checked that they **sum** to zero, so `[Ca2+].[Cl-]`
counted as charge-balanced. The corpus is 4,788/4,790 exactly (+1,−1) with net
charge zero, so `net_charge_zero` is the criterion the data supports; it is now
reported separately and folded into `plausible`.

Every plausibility metric is calibrated on the real SMILES of each evaluation set
(the `real_*` arms, 0.991–0.996 throughout) and probed with deliberate junk
(0.167). A plausibility score on which real ILs do not land near 1.0 would mean
the metric is wrong, not the model — the same logic as the AE ceiling in §5.

### 6.6 Flow matching vs DDPM — same latent cloud, same everything else

Run 18 Sep 21:37 → 19 Sep 00:19, `scripts/run_il_ddpm.sh`, CPU only.

The staged architecture makes this comparison unusually clean. Stages 0–3 are
shared and frozen — same pretrained decoder, same z cloud, same splits — so
**only Stage 4's objective and sampler change.** `fm/ddpm.py` exposes the same
`loss(x1, y, x0)` signature as `FlowMatching`, so it drops into the same
`Trainer`, `EMA` and `VectorDataset`. The network is the identical
`VelocityMLP`, reinterpreted to predict ε instead of a velocity: same 2.6M
parameters, same FLOPs per evaluation.

Matched by construction: AdamW at lr 1e-3, warmup + cosine, grad clip 1.0, EMA
0.999, 8,000 steps at batch 128. Training FLOPs agree to within 2% (15.8–16.3 T)
and measured training time to within 15% (248–286 s), so **the entire efficiency
difference is sampling NFE.**

DDPM hyperparameters are the textbook ones (Ho et al. 2020): T=1000,
ε-prediction, linear β from 1e-4 to 0.02.

> **Time runs the other way here.** In `fm/paths.py`, t=0 is noise and t=1 is
> data. In `fm/ddpm.py`, index i=0 is *data* and i=T−1 is *noise* — the standard
> diffusion convention, and the usual source of sign confusion when the two are
> read side by side. The net sees `(i+1)/T`, so larger still means noisier, and
> at T=1000 `SinusoidalTimeEmbedding`'s internal ×1000 hands it exactly the
> integer timestep.

#### Result: flow matching wins on quality *and* cost

| d | FM heun-50 | DDPM ancestral | DDIM-50 | gap (FM − anc) | ceiling |
|---|---|---|---|---|---|
| 16 | **0.682** | 0.634 | 0.621 | +0.048 (+7.5%) | 0.689 |
| 32 | **0.642** | 0.615 | 0.576 | +0.028 (+4.6%) | 0.682 |
| 64 | **0.636** | 0.559 | 0.538 | +0.077 (+13.8%) | 0.710 |
| 128 | **0.569** | 0.406 | 0.384 | +0.163 (**+40.1%**) | 0.668 |

| | NFE/sample | sample FLOPs | sample time, 2000 | train time |
|---|---|---|---|---|
| FM heun-50 | 100 | 514–531 M | 6–9 s | 248–286 s |
| DDPM ancestral | **1000** | **5.14–5.31 G** | **71–83 s** | 258–281 s |
| DDIM-50 | 50 | 257–265 M | 3–4 s | 258–281 s |

**Flow matching beats DDPM ancestral at every `d`, at one tenth the sampling
cost.** This is not a close call at the top end: at d=128 flow matching is 40%
better while doing 10× less work at generation. There is no accuracy-for-compute
trade here to negotiate — one method dominates the other on both axes.

DDIM-50 is the interesting third point. It halves flow matching's NFE, but at
d=16 it scores 0.621 against 0.682, so flow matching still wins on quality at
twice the cost. **DDIM is the right choice only if sampling cost is the binding
constraint**; otherwise flow matching's 100 NFE is the better buy.

Note also how the gap tracks §6.3's finding: DDPM degrades *faster* with `d`
than flow matching does (0.634 → 0.406, a 36% fall, versus 0.682 → 0.569, 17%).
Whatever makes a bigger latent harder to model, discrete-time diffusion suffers
from it more.

#### Sliced W₂ does not predict plausibility — a live demonstration

At d=16, DDPM ancestral has a **better** `test_ion` W₂ than flow matching (0.122
vs 0.130) and an identical `test` W₂ (0.093), yet decodes to *worse* molecules
(0.634 vs 0.682). The two metrics disagree in sign.

This is exactly the blind spot documented in `fm/il_eval.sliced_w2`: it is
insensitive to mode dropping inside a matched envelope, and it only ever sees
1-D projections. DDPM's samples land in the right overall region of latent space
— the projections agree — but in worse *places within* it, and only decoding
reveals that. At d=128 the two metrics finally agree (W₂ 0.137 vs 0.171,
plausibility 0.569 vs 0.406), which is what a genuinely large gap looks like.

**The lesson is the one §6.4 already states, now with a concrete case: judge
quality with a domain metric on decoded samples.** W₂ on latents isolates the
flow, which is valuable for attribution, but it is not a quality score and a
small difference in it means very little.

#### The OOD gap is a property of the latent space, not the generative model

`sw2_ood` sits at 0.463–0.521 for *every* model and sampler, with no systematic
ordering. Flow matching does not place more mass near the ILThermo latents than
DDPM does, and neither comes close to the in-distribution columns (~0.13).

That is worth stating plainly: **swapping the generative model does not improve
OOD behaviour at all.** Where the OOD latents sit is decided by the frozen
encoder and the fine-tuned `MLP_down`, upstream of Stage 4 entirely. If the OOD
gap is the thing to fix, Stage 4 is the wrong place to work.

Recovery of unseen `test_ion` cations is similarly flat: 0.041–0.079 across all
twelve rows, with flow matching modestly ahead at d ≤ 64 (0.058–0.079 vs
0.041–0.058) and behind at d=128. Those are single-run differences of a few
molecules and should not be read as a result.

#### Two standard-looking DDPM settings diverge — verify before trusting

Checked on a 2-D ring first, the way `sanity_check.py` validates flow matching.
Two configurations that look entirely ordinary in a paper **blow up**:

| setting | modes | mean radius | std vs true | sample |
|---|---|---|---|---|
| FM heun-50 | 8/8 | 98.4% | **98.6%** | 1.8 s |
| DDPM linear, ancestral | 8/8 | 93.6% | 94.2% | 18.4 s |
| **DDPM cosine, ancestral** | 8/8 | 3186% | **41,373%** | 20.8 s |
| DDPM cosine, ancestral, clip 4 | 8/8 | 90.4% | 90.5% | 30.2 s |
| **DDIM-50, no clip** | 8/8 | 118% | **357%** | 0.9 s |
| DDIM-50, clip 3 | 8/8 | 95.5% | **96.5%** | 0.9 s |

One cause behind both: sampling divides by `sqrt(abar_i)` to recover `x0_pred`.
The cosine schedule drives `abar_T` to **2.4e-9**, so that division is by 5e-5
and any ε error is amplified 2×10⁴. Ancestral sampling damps this because the
posterior weight on `x0_pred` is tiny at high noise; DDIM leans on `x0_pred`
directly, so it diverges even under the linear schedule.

Clamping the predicted `x0` fixes both, which is why `DDPM.sample` is written in
the predict-x0 form — algebraically identical to the ε form, but it exposes the
one knob that matters. **Use the linear schedule.** `clip_x0=3` is a verified
no-op for linear + ancestral (identical to three decimals), so a single flag
serves both samplers without altering the standard baseline.

Note the ordering: DDIM-50 with clipping (96.5%) beats ancestral (94.2%) on the
toy *and* is 23× cheaper, yet on real ILs ancestral wins at every `d`. A 2-D
sanity check proves an implementation correct; it does not rank methods.

#### What this comparison cannot say

- **One seed, one split draw.** Every row is `seed=0`, `--split-seed 0`.
- **8,000 steps for both.** Fixed budget, not converged-to-best. DDPM is
  sometimes argued to need longer; if so, part of this gap is budget, not method.
- **T=1000 untuned.** It is the standard value, not a searched one, and the
  `d` sweep in §6.3 showed this pipeline is sensitive to defaults chosen that way.
- **Reconstruction is not a differentiator.** The decoder is frozen and shared,
  so the `ceiling` and `ae_recon` arms are identical across both models by
  construction. They appear in the table as a shared reference line, not a
  comparison.

### 6.7 Why `d` can be this small — the data is ~8-dimensional

§6.3 found generation improving as `d` fell, which raises an obvious objection:
how can a 36-token molecule fit in 16 dimensions, let alone 8? And is the answer
just that 4,790 pairs is too little data to need more? Both are measurable.

```bash
.venv/bin/python scripts/il_intrinsic_dim.py
```

#### The encoder's output cloud has an effective rank of 8.2

Participation ratio, `(Σλ)² / Σλ²`, of the ChemBERTa embeddings that `MLP_down`
receives — an upper bound on what `z` can usefully carry:

| variance captured | PCs needed |
|---|---|
| 50% | **3** |
| 80% | 12 |
| 90% | 27 |
| 95% | 57 |

**PR = 8.2 out of 768**, with the leading PC alone holding 27.5%. So `d=8` is
roughly the natural width of this data, and `d=64` was carrying about 8
dimensions of chemistry plus ~56 of noise.

#### It is not sample-limited, and that is directly testable

If 4,790 pairs were too few to reveal the manifold, effective rank would still
be climbing with `n`. It is flat:

| n | 200 | 500 | 1000 | 2000 | 3447 | 4790 |
|---|---|---|---|---|---|---|
| PR | 8.39 | 8.11 | 7.98 | 8.28 | 8.23 | **8.21** |

**200 molecules already reveal the same dimensionality as all 4,790.** More ILs
drawn from the same families would sample this manifold denser, not wider.

The control that isolates chemistry from sample count: MOSES drug-like
molecules, same frozen encoder, **matched n=4,790** → **PR = 18.95**, 2.3x
higher. Only the molecules differ.

#### Why ionic liquids are this low-rank

They are a structurally homogeneous family. Ion identity alone explains most of
the variance — **anion 61.9%, cation 64.9%** — and there are only 374 distinct
anions, one of which covers 913 pairs. What actually varies is a handful of
things: cation scaffold (imidazolium 34%, ammonium 29%, pyridinium 10%), alkyl
chain length, anion class, degree of fluorination. PC1 correlates r=+0.56 with
heavy-atom count, so one of those degrees of freedom is plain molecular size.

Mean-pooling contributes as well: averaging token states discards positional
structure and biases toward low rank. How much of the 8.2 is the chemistry and
how much is the pooling is an open question, and a CLS or attention-pooled
encoder would answer it.

#### What this explains

§6.3's trend is not "small latents are mysteriously better." Above roughly 8–27
dimensions the extra coordinates carry almost no chemical variance, so they
carry **noise** — and Stage 4 then has to spend capacity modelling that noise.
That is the mechanism behind fm plausibility falling monotonically with `d`
while reconstruction rises slightly: the decoder gains a little from the extra
width, and the flow loses more.

It also explains why §6.6's DDPM degrades faster with `d` than flow matching
does. Both are being handed the same growing pile of noise dimensions.

#### Where data *does* limit this pipeline

Not the latent width — the decoder. Ceiling plausibility is 0.67–0.71 and exact
reconstruction ~0.09 at every `d`: the autoencoder never became high-fidelity.
It emits IL-*shaped* strings with the right family and charge rather than exact
molecules, which is a 3,447-example problem against a 2.5M-parameter decoder.

Track B is the evidence for what more data does to the optimum: MOSES at 200k
(45x more) showed the **opposite** `d` trend, token accuracy rising monotonically
to d=128. So optimal `d` does scale with dataset size — but for ILs it would
move up only if new data brought genuinely new chemistry, not more
imidazolium/NTf2 variants.

#### The OOD set is shifted within the manifold, not off it

Sharpening §6.6, which found `sw2_ood` ~0.47 for every model:

| | train | ood |
|---|---|---|
| energy inside train's top-64 PCs | 0.956 | 0.954 |
| mean \|z\| along top-16 PCs | 0.788 | 0.802 |

Statistically identical — the OOD set is **not** off-manifold. What it is instead
is a **mean shift within** the manifold, 0.30–0.58 SD along the leading PCs.
Aggregate radius hides that, because a 0.3 SD shift barely moves E|z|.

A location shift is exactly what sliced W₂ detects well, since it moves the
quantile functions along most random projections. So the large `sw2_ood`
alongside in-distribution-looking subspace statistics is not a contradiction —
the two measurements are sensitive to different things, and together they locate
the difference precisely. It confirms §6.6's conclusion: the OOD gap originates
upstream of Stage 4, so no change to the generative model addresses it.

#### The prediction this makes

`d=8` at or near the optimum, `d=4` starting to lose, since 4 < 8.2 is genuine
information loss. The caveat that could falsify it: participation ratio is a
**linear** measure and `MLP_down` is nonlinear, so 8.2 upper-bounds what a
nonlinear encoder needs rather than flooring it.

**§6.8 falsified this as stated.** Plausibility kept rising to `d=4`; the loss
appeared in *reconstruction* instead, and began at `d=8`. The measurement here
holds — the prediction named the wrong metric.

### 6.8 `d` = 8 and 4 — the prediction was wrong, and the way it failed is the result

Run 19 Sep 08:14 → 11:15, `scripts/run_il_lowdim.sh`. §6.7 predicted `d=8` at or
near the optimum and `d=4` starting to lose. **Plausibility says otherwise:**

| d | gate | plaus | uniq | **n_unique** | novel_cat | ceiling | sw2 test_ion |
|---|---|---|---|---|---|---|---|
| **4** | 0.8860 **FAIL** | **0.817** | 0.366 | **598** | 0.187 | 0.831 | 0.173 |
| **8** | 0.8972 **FAIL** | 0.695 | 0.567 | 788 | 0.224 | 0.710 | 0.154 |
| 16 | 0.9062 PASS | 0.682 | 0.612 | **835** | 0.287 | 0.689 | 0.130 |
| 32 | 0.9124 PASS | 0.642 | 0.612 | 786 | 0.272 | 0.682 | 0.138 |
| 64 | 0.9180 PASS | 0.636 | 0.658 | **838** | 0.306 | 0.710 | 0.129 |
| 128 | 0.9172 PASS | 0.569 | 0.649 | 738 | 0.287 | 0.668 | 0.137 |

Plausibility rises monotonically all the way down to `d=4`, which scores **0.817
— the highest of any configuration in this study.** Taken alone that would say
"keep shrinking."

#### It is a mirage, and §6.2 already taught us how to spot it

Two things give it away.

**Both `d=8` and `d=4` fail the Stage-2 gate** (0.8972 and 0.8860 against 0.90).
By the rule in §8 their numbers are suspect before they are even read.

**The ceiling arm moves with the model.** At `d=4` the ceiling's plausibility is
the *highest* in the study (0.831) while its uniqueness is the *lowest*
(0.615 against 0.848–0.879 elsewhere), on 220 distinct molecules against
252–269. The ceiling is fed *real encoded latents* — so this is not the flow
being better, it is **the decoder itself collapsing onto a narrow set of easy,
common ILs.** Mode concentration in the generated samples confirms it: the top-5
modes are 10.2% of `d=4`'s output against 4.0% at `d=16`.

This is the `ar_random` pattern from §6.2 in milder form — high validity,
collapsed uniqueness, ceiling equally high — and it is why §5 insists uniqueness
is read as a mode-collapse detector rather than a quality score. **`n_unique` is
the honest headline:** `d=4` yields 598 distinct plausible molecules where
`d=16` yields 835.

#### The information floor is real — it just shows up in reconstruction

§6.7 put the manifold's effective rank at 8.2 and predicted the loss would
appear at `d=4`. The loss *did* appear at low `d`; it surfaced in the **gate**
rather than in plausibility, and it began at `d=8`, not `d=4`.

The distinction worth keeping: participation ratio measures how many dimensions
**span** the data's variance. Exact reconstruction is a higher bar — it requires
enough capacity to **index individual molecules**, not merely to cover the
manifold they lie on. 8 dimensions suffice to place a point on the IL manifold
and quite reasonably to model its *distribution*; they do not suffice to say
*which* molecule it is. That is exactly the gap between `sw2_test_ion` (fine at
`d=8`) and `val_token_acc` (below threshold at `d=8`).

So §6.7's measurement was sound and its prediction was mis-specified: it named
the wrong metric. The manifold dimension bounds where the distribution lives,
not how much room reconstruction needs above it.

#### Revised recommendation: `d=16`

It is the smallest gate-passing configuration, has the highest plausibility of
any gate-passing configuration (0.682), and ties `d=64` for the most distinct
plausible molecules (835 vs 838) while beating it on plausibility. **The optimum
is a floor set by reconstruction, not the downward trend §6.3 extrapolated.**

#### The flow-matching advantage vanishes at low `d`

Extending §6.6 to the new points, flow matching minus DDPM ancestral:

| d | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| gap | +0.020 | +0.014 | +0.048 | +0.028 | +0.077 | **+0.163** |

At `d=4` the three generative models are within 0.021 of each other
(FM 0.817, DDIM 0.811, ancestral 0.796). **The choice of generative model stops
mattering once the latent is small enough**, which supports §6.6's proposed
mechanism directly: flow matching's advantage is in tolerating noise dimensions,
so removing the noise dimensions removes the advantage. If you operate at small
`d`, take DDIM-50 and its 50 NFE — at `d=4` it costs half of flow matching's
sampling budget for 0.006 less plausibility.

### 6.9 Raising the decoder ceiling — the constraint was data, not capacity

§6.8 ended by saying every route onward runs through the decoder ceiling. This
tests that, and the motivating number is stark: at `d=16`, flow matching had
closed **98.2%** of the prior→ceiling gap. Roughly 2% of headroom remained, so
the FM-vs-DDPM comparison at the best operating point could not discriminate
between two models no matter how different they were.

Run 19 Sep 13:44 → 17:41, `scripts/build_il_enumerated.py` + `scripts/run_il_enum.sh`.

#### What was added

The observed pairs are 0.57% of the cation × anion grid. Every unobserved
combination still pairs two **real** ions from measured ILs, so it is valid
supervision for the decoder's actual job (`z → SMILES`) even though nobody has
made that particular salt. Enumerating from **train ions only** (1,706 × 333 =
568,098 available) gives **102,831 pairs — 30× the decoder's previous
supervision** — with two asserts that no held-out pair and no `test_ion` cation
can leak in.

The flow still trains on the 4,790 observed pairs alone. The observed set is not
a random slice of the grid; it is what chemists actually made, which encodes
synthesizability and stability, and that structure is the thing worth modelling.

#### Result

| config | gate | exact | prior | model | **ceiling** | % gap | uniq | **n_unique** |
|---|---|---|---|---|---|---|---|---|
| BASE FM d16 | 0.9062 | 0.038 | 0.301 | 0.682 | 0.689 | 98.2% | 0.612 | 835 |
| BASE DDPM d16 | 0.9062 | 0.038 | 0.301 | 0.634 | 0.689 | 85.9% | 0.585 | 742 |
| **ENUM FM d16** | **0.9464** | **0.131** | 0.491 | **0.851** | **0.903** | 87.4% | **0.781** | **1329** |
| ENUM DDPM d16 | 0.9464 | 0.131 | 0.491 | 0.806 | 0.903 | 76.5% | 0.774 | 1248 |
| BASE FM d64 | 0.9180 | 0.064 | 0.293 | 0.636 | 0.710 | 82.4% | 0.658 | 838 |
| **ENUM FM d64** | **0.9541** | **0.200** | 0.488 | 0.809 | 0.882 | 81.5% | 0.815 | 1319 |
| ENUM DDPM d64 | 0.9541 | 0.200 | 0.488 | 0.757 | 0.882 | 68.3% | 0.805 | 1219 |

**The ceiling moved 0.689 → 0.903 at `d=16`.** Flow-matching plausibility rose
0.682 → **0.851**, and distinct plausible molecules 835 → **1,329 (+59%)**.
Uniqueness rose at the same time (0.612 → 0.781), so this is not the
fidelity-for-diversity trade of §6.3 — both moved together.

**The decoder was data-limited, not capacity-limited.** The same 1.0M-parameter
single-layer GRU, given 30× the supervision, improved by +0.057 on train *and*
**+0.045 on `test_ion`** reconstruction. It generalizes rather than memorizing
the enumerated pairs, so a bigger decoder or a transformer was not what was
missing. Exact reconstruction — the capacity signal §6.8 identified — went
0.038 → 0.131 at `d=16` and 0.064 → **0.200** at `d=64`, roughly 3×.

**Phase 2b held the latent in place.** `sw2_test_ion` is 0.130 → 0.134 (`d=16`)
and 0.129 → 0.138 (`d=64`), essentially unchanged, so the enumerated corpus
raised decoder capability *without* dragging the frozen latent toward the
uniform product distribution of ions. That was the specific risk the 2a/2b split
was designed against, and it is the measurement that says the design worked.

#### Two earlier conclusions survive the higher ceiling

**Flow matching still beats DDPM**, and the gap is close to unchanged:

| | d=16 | d=64 |
|---|---|---|
| baseline | +0.048 | +0.077 |
| enumerated | +0.044 | +0.052 |

So §6.6's ranking was not a ceiling artifact. What *did* change is that the
comparison finally has room: at baseline `d=16` flow matching sat at 98.2% of
gap with nowhere to go, and now it sits at 87.4% against DDPM's 76.5%.

**`d=16` still beats `d=64`** — 0.851 vs 0.809 — even though `d=64` reconstructs
better (exact 0.200 vs 0.131). §6.3's finding is therefore *not* an artifact of a
weak decoder: the extra latent width still costs more in density estimation than
it returns in decodability, exactly as §6.7's effective-rank argument predicts.

#### What did not improve

`sw2_ood` rose 0.467 → 0.536 at `d=16`. A better decoder pulled the latent
slightly *further* from the ILThermo cloud. Consistent with §6.7: the OOD shift
originates in the frozen encoder and `MLP_down`, upstream of everything changed
here, and no amount of decoder work addresses it.

The prior arm also rose sharply (0.301 → 0.491), which is worth noting when
reading "% of gap" across the two regimes — a stronger decoder makes even
unstructured latents decode to something plausible, so the denominator shrank.
Absolute plausibility and `n_unique` are the safer cross-regime comparisons.

---


---


---


---




---

## 7. Dead ends — do not repeat these

**ChemBERTa-77M-MLM silently destroys charges.** `C[N+](C)(C)C` → `CN(C)(C)C`,
`[Cl-]` → `C-`, `[C@@H]` → `C`. Its vocab *contains* `[N+]`/`[B-]`, but not `[`,
`]`, `+`, or `@`; the normalizer strips brackets before lookup, so those tokens
are unreachable and the rest falls to character level. It emits **zero UNK**, so
it fails completely silently.

It cannot be rescued by manual re-tokenization: the bracket tokens' embeddings
have norm **2.40 ± 0.089** versus **3.87 ± 0.51** for characters known to be
used — the tight low-norm cluster of untrained weights. They were never trained,
because the same normalizer stripped them during pretraining.

**Use `seyonec/ChemBERTa-zinc-base-v1`**: 44.1M params, hidden 768, round-trips
charges and stereochemistry, 0 UNK on IL pairs. Costs 24 mol/sec vs 244.
*Always round-trip a charged SMILES through any new tokenizer before trusting it.*

**IL pair data: ILThermo + PubChem tops out near 350 pairs.** ILThermo names ~2,072
pure ILs but publishes no structures, and PubChem resolves only ~17% of them —
the common ILs resolve, the long tail does not. Compositional name-splitting
(cation + anion separately) does not rescue it either: **523 of 785 cation names
appear exactly once**, so there is no popular-ion shortcut.

**Use Zenodo record 3251643**, which ships `CA.smi` (ion → SMILES) plus one
observed-pair file per measured property. Unioning all 12 property files yields
**4,790 validated pairs** (2,253 cations, 374 anions), 2,212 with melting points.
Built by `scripts/build_il_dataset_v2.py`.

---

## 8. Running it

```bash
.venv/bin/python scripts/il_embed.py --moses-n 200000      # Stage 0, ~2.8 h
./scripts/run_il_overnight.sh                              # Stages 1-5, d=64 configs
./scripts/run_il_sweep.sh                                  # 2x2 cell D + the d sweep
.venv/bin/python scripts/build_il_ood.py                   # the ILThermo OOD test set
.venv/bin/python scripts/il_embed.py --only ood            # ~10 s
./scripts/run_il_eval.sh                                   # split protocol, all d (~80 min)
./scripts/run_il_ddpm.sh                                   # flow matching vs DDPM (~2.7 h)
.venv/bin/python scripts/il_intrinsic_dim.py               # effective rank of the latent
./scripts/run_il_lowdim.sh                                 # d = 8 and 4 (~3 h, needs pretrains)
.venv/bin/python scripts/build_il_enumerated.py --n 100000  # 103k train-ion pairs
.venv/bin/python scripts/il_embed.py --only enum           # ~110 min
./scripts/run_il_enum.sh                                   # raised-ceiling FM vs DDPM (~4 h)
.venv/bin/python scripts/analyze_il.py                     # figures + tables
```

Measured CPU timings (12 cores, no GPU), from the runs in §6:

| step | cost |
|---|---|
| embed, 24 mol/s | 200k MOSES ≈ 2.3 h |
| AR pretrain, 217k × 2 ep | 41–67 min |
| NAR pretrain, 217k × 2 ep | 28 min |
| fine-tune, 4790 × 40 ep | 9–20 min |
| FM or DDPM training, 8000 steps | ≈ 4.5 min (248–286 s, equal for both) |
| sampling 2000, FM heun-50 (100 NFE) | 6–9 s |
| sampling 2000, DDPM ancestral (1000 NFE) | 71–83 s |
| sampling 2000, DDIM-50 (50 NFE) | 3–4 s |
| generate + score, 3 arms × 2000 | ≈ 5 min |

The AR pretrain range is not a `d` effect — `d=32` took 66.5 min against
`d=128`'s 47.6 min. It is contention from other work on the same box. Budget by
the top of the range. One full `d` point, pretrain through scoring, is ~70–95
min; the whole four-point sweep plus the NAR/random cell ran 12:01→16:04.

**Gate:** Stage 2 reports `val_token_acc` against a 0.90 threshold. Below it, the
latent is suspect and Stage 4/5 numbers should not be trusted. The run continues
rather than aborting (a full night is available either way), but `gate_pass` is
recorded in `results.jsonl` — check it before reading any result.

**Cost of a `d` sweep.** Pretraining is tied to `d` — both `MLP_down`'s output
width and the decoder's conditioning depend on it — so a checkpoint from one `d`
cannot be loaded at another and every point needs its own Stage 1. The four-point
sweep in §6.3 cost ~4 h of CPU for that reason. It ran fine without a GPU; a GPU
would mainly buy the ability to extend it.

**What to run next**, cheapest first:

1. **More enumerated data, and more 2a epochs.** §6.9 moved the ceiling
   0.689 → 0.903 with 103k pairs and only 2 epochs of phase 2a, and nothing
   suggests that saturated. 568k combinations are available from train ions;
   the obvious next point is 300k pairs or 4-6 epochs, measured the same way.
2. **Decoder sampling temperature.** `ARDecoder.generate` is greedy argmax
   (`il_pipeline.py:75`), which is a known mode-collapse source in AR text
   models. Now that `--save-ae` exists, temperature and nucleus sampling are
   testable without retraining Stage 2.
2. **More FM steps at small `d`.** All runs used 8,000 steps regardless of `d`.
   A 16-dim cloud may be converged there while a 128-dim one is not, which would
   mean §6.3's trend partly measures a fixed step budget rather than intrinsic
   difficulty. Re-run `d=128` at 32k steps to separate the two.
3. **A second seed.** Every number in §6 is a single run at `seed=0` and one
   `--split-seed 0`. The `d` trend is monotone across four points, so it is
   unlikely to be noise, but the uniqueness step between `d=32` and `d=64` rests
   on one run per point, and §6.5's overfitting ratio rests on one split draw.
4. **Attack the flow's overfitting**, now that §6.5 has made it measurable: the
   velocity field sits 2.3-3.1x closer to train than to held-out latents. More
   FM steps will not help and may hurt; weight decay, a smaller `--fm-width`,
   or augmenting the z-cloud are the things to try, judged on the `test_ion`
   W2 column rather than the train one.
5. **Give DDPM a longer budget.** §6.6 held both models to 8,000 steps. If the
   gap narrows at 32k, part of it was budget rather than method; if it holds,
   the result is about the objective.
6. **Exact NLL via the change-of-variables formula.** `d log p_t/dt = -div v`,
   integrated backward from a real latent. It is the one principled scalar for
   the flow alone, it needs no new data, and at `d=16` the divergence can be
   taken exactly with 16 autograd passes per step. Report it on `test_ion`.

## 9. Files

| file | role |
|---|---|
| `fm/il_pipeline.py` | `MLP_down`, `MLP_up`, `ARDecoder`, `NARDecoder`, `LatentAE` |
| `scripts/build_il_dataset_v2.py` | Zenodo → 4,790 validated pairs |
| `scripts/il_embed.py` | Stage 0 |
| `scripts/il_pretrain.py` | Stage 1 |
| `scripts/il_finetune_fm.py` | Stages 2–5 |
| `scripts/run_il_overnight.sh` | orchestrator, `d=64` configs |
| `scripts/run_il_sweep.sh` | orchestrator, NAR/random cell + `d` sweep |
| `fm/ddpm.py` | `DDPM` — the diffusion baseline, swappable with `FlowMatching` |
| `fm/il_eval.py` | splits, chemical plausibility, sliced W2, novelty, recovery |
| `scripts/build_il_ood.py` | ILThermo/PubChem leftovers → the 227-pair OOD test set |
| `scripts/run_il_eval.sh` | orchestrator, split protocol (Stages 2-5 only) |
| `scripts/run_il_ddpm.sh` | orchestrator, flow matching vs DDPM |
| `scripts/run_il_lowdim.sh` | orchestrator, `d` = 8 and 4 |
| `scripts/build_il_enumerated.py` | 103k train-ion pairs for decoder training, split-safe |
| `scripts/run_il_enum.sh` | orchestrator, raised-ceiling FM vs DDPM |
| `scripts/il_intrinsic_dim.py` | effective rank of the embedding cloud, and whether it is sample-limited |
| `scripts/analyze_il.py` | figures and summary tables (dedups `results.jsonl` by tag, splits schema 1 / 2) |
| `runs/il/results/results.jsonl` | one JSON line per config |
| `runs/il/results/fig_il_arms.png` | three arms per config, gate failures shaded |
| `runs/il/results/fig_il_ceiling.png` | how much of prior→ceiling the flow closed |
| `runs/il/results/fig_il_dsweep.png` | the `d` sweep |
| `runs/il/results/fig_il_generalization.png` | held-out W2, reconstruction and plausibility by split |
| `runs/il/results/fig_il_fm_vs_ddpm.png` | quality, cost-vs-quality and latent fit for both models |
