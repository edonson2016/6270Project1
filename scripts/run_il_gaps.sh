#!/usr/bin/env bash
# Gap-filling runs demanded by the write-up's own rubric. Each one closes a
# specific hole, named here so the reason survives the run:
#
#  1. SEEDS on the headline FM-vs-DDPM contrast. Section 4.2 rested on ONE seed
#     per model, so it could report a direction but no uncertainty. Three seeds
#     each gives a spread to quote.
#  2. The DDIM cell at the CURRENT decoder. The cost-matched diffusion arm only
#     ever existed on the older SMILES autoencoder, so the SELFIES-era comparison
#     had no 50-NFE diffusion row. --sampler ancestral,ddim emits both from ONE
#     trained model, which is also what keeps them fairly matched.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
AE=runs/il/ae/ae_sf_d16.pt
PRE=runs/il/pretrain_sf_d16/pretrained.pt
COMMON="--pretrained $PRE --tokenizer selfies --max-len 96 --latent-dim 16 --decoder ar --load-ae $AE"

echo ""; echo "=== GAPS RUN START $(date) ==="

# --- 1. FM seeds (gauss + independent = the section 4.2 baseline arm)
for S in 1 2; do
  T="sf_base_s$S"
  echo ""; echo "=== $T (FM, gauss, independent, seed $S) ==="
  timeout 7200 $PY scripts/il_finetune_fm.py $COMMON \
    --generative fm --sampler heun --seed "$S" --tag "$T" 2>&1 | tee "runs/il/s25_$T.log"
  echo "--- exit $? at $(date) ---"
done

# --- 2. DDPM seeds, both samplers from one trained model
for S in 0 1 2; do
  T="sf_ddpm_s$S"
  echo ""; echo "=== $T (DDPM T=1000 linear, ancestral+ddim, seed $S) ==="
  timeout 7200 $PY scripts/il_finetune_fm.py $COMMON \
    --generative ddpm --ddpm-T 1000 --ddpm-schedule linear \
    --sampler ancestral,ddim --clip-x0 3.0 --seed "$S" --tag "$T" 2>&1 | tee "runs/il/s25_$T.log"
  echo "--- exit $? at $(date) ---"
done

# --- 3. Recipe seeds, so the innovation's spread is measured on the FINAL config
for S in 1 2; do
  T="sf_sig010_s$S"
  echo ""; echo "=== $T (FM, full, ot, sigma 0.10, seed $S) ==="
  timeout 7200 $PY scripts/il_finetune_fm.py $COMMON \
    --generative fm --sampler heun --source full --coupling ot --sigma-min 0.10 \
    --seed "$S" --tag "$T" 2>&1 | tee "runs/il/s25_$T.log"
  echo "--- exit $? at $(date) ---"
done

echo ""; echo "=== GAPS RUN DONE $(date) ==="
