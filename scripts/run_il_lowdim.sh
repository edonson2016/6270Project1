#!/usr/bin/env bash
# Bracket the latent-dimension optimum below d=16.
#
# Motivation (docs §6.7): the participation ratio of the ChemBERTa IL embedding
# cloud is 8.2 of 768, and it is flat from n=200 to n=4790, so it is a property
# of the chemistry rather than the sample size. Prediction to be tested here:
#   d=8  at or near the optimum
#   d=4  starts losing, because 4 < 8.2 is genuine information loss
# The caveat that could falsify it: participation ratio is a LINEAR measure and
# MLP_down is nonlinear, so 8.2 is an upper bound on what is needed, not a floor.
#
# Each d needs its own pretrain -- MLP_down's output width and the decoder's
# conditioning both depend on it -- and that dominates the cost. DDPM at these
# points is therefore nearly free on top, so §6.6's comparison is extended too.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p runs/il/results
echo "=== LOW-D RUN START $(date) ==="

for D in 8 4; do
  echo ""; echo "=== STAGE 1: pretrain AR decoder (d=$D) ==="
  timeout 14400 $PY scripts/il_pretrain.py --latent-dim $D --decoder ar --epochs 2 \
    --out runs/il/pretrain_ar_d$D 2>&1 | tee runs/il/stage1_ar_d$D.log
  echo "--- exit $? at $(date) ---"

  echo ""; echo "=== FM  d=$D (heun-50, 100 NFE) ==="
  timeout 7200 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_ar_d$D/pretrained.pt \
    --latent-dim $D --decoder ar --generative fm --sampler heun \
    --tag gen_fm_d$D 2>&1 | tee runs/il/s25_gen_fm_d$D.log
  echo "--- exit $? at $(date) ---"

  echo ""; echo "=== DDPM d=$D (T=1000 linear; ancestral + DDIM-50) ==="
  timeout 10800 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_ar_d$D/pretrained.pt \
    --latent-dim $D --decoder ar --generative ddpm --ddpm-T 1000 --ddpm-schedule linear \
    --sampler ancestral,ddim --clip-x0 3.0 \
    --tag gen_ddpm_d$D 2>&1 | tee runs/il/s25_gen_ddpm_d$D.log
  echo "--- exit $? at $(date) ---"
done

echo ""; echo "=== LOW-D RUN DONE $(date) ==="
$PY scripts/analyze_il.py 2>&1 | tail -30
