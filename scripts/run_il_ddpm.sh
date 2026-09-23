#!/usr/bin/env bash
# Flow matching vs DDPM on the same latent cloud, same splits, same everything else.
#
# The comparison is clean because Stages 0-3 are shared and frozen: identical
# pretrained decoder, identical z cloud, identical splits. Only Stage 4's
# objective and sampler change, so any difference is attributable to those.
#
# Matched by construction: same VelocityMLP (width 384, depth 4, ~2.6M params),
# same AdamW at lr 1e-3, same warmup+cosine schedule, same grad clip, same EMA
# decay 0.999, same 8000 steps at batch 128. Training FLOPs are therefore equal
# to within 2%; the whole efficiency story is sampling NFE.
#
# DDPM hyperparameters are the textbook ones (Ho et al. 2020): T=1000,
# eps-prediction, linear beta 1e-4 -> 0.02. Verified on a 2-D toy first:
#   - linear + ancestral recovers 8/8 modes at 93.6% of true spread
#   - cosine + ancestral DIVERGES (41,000% of true spread) because abar_T ~ 2e-9
#     makes x0_pred divide by 5e-5; it needs clip_x0 and is still worse
#   - DDIM-50 without clipping also diverges (360%); with clip_x0=3 it is the
#     best DDPM setting tested and 23x cheaper than ancestral
# clip_x0=3 is a verified no-op for linear+ancestral (identical to 3 decimals),
# so one flag serves both samplers without altering the standard baseline.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p runs/il/results
echo "=== FM vs DDPM START $(date) ==="

for D in 16 32 64 128; do
  CK=runs/il/pretrain_ar_d$D/pretrained.pt
  [ "$D" = "64" ] && CK=runs/il/pretrain_ar/pretrained.pt
  if [ ! -f "$CK" ]; then echo "MISSING $CK -- skipping d=$D"; continue; fi

  echo ""; echo "=== FM  d=$D (heun, 50 steps = 100 NFE) ==="
  timeout 7200 $PY scripts/il_finetune_fm.py --pretrained "$CK" --latent-dim $D \
    --decoder ar --generative fm --sampler heun \
    --tag gen_fm_d$D 2>&1 | tee runs/il/s25_gen_fm_d$D.log
  echo "--- exit $? at $(date) ---"

  echo ""; echo "=== DDPM d=$D (T=1000 linear; ancestral 1000 NFE + DDIM 50 NFE) ==="
  timeout 10800 $PY scripts/il_finetune_fm.py --pretrained "$CK" --latent-dim $D \
    --decoder ar --generative ddpm --ddpm-T 1000 --ddpm-schedule linear \
    --sampler ancestral,ddim --clip-x0 3.0 \
    --tag gen_ddpm_d$D 2>&1 | tee runs/il/s25_gen_ddpm_d$D.log
  echo "--- exit $? at $(date) ---"
done
echo ""; echo "=== FM vs DDPM DONE $(date) ==="
