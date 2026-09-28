#!/usr/bin/env bash
# AUTOGUIDANCE PHASE 0 -- train the ladder of guiding models.
#
# Autoguidance samples from  v~ = (1+w)*v1 - w*v0,  extrapolating away from a
# deliberately degraded version of the SAME model. Writing v1 = v* + e1 and
# v0 = v* + e0, the guided error is  e1 + w*(e1 - e0), so guidance only helps
# when <e1, e1-e0> < 0 -- that is, when the weak model fails in the same
# DIRECTION as the strong one, just harder. A weak model that fails differently
# makes things strictly worse at every w.
#
# So every model here differs from runs/il/_fm_enum_fm_d16 in data volume and
# training length and NOTHING else: same autoencoder checkpoint, same frozen
# latent geometry, same architecture, same corpus and splits. The Standardizer
# is fit on the full 3,447-point train cloud even for the subset runs, so all
# six fields share one coordinate frame and v1-v0 is a meaningful vector.
#
# The two controls exist because "less data + shorter training" confounds two
# axes; wtime and wdata separate them.
#
#   w05    172 latents  2,000 steps    ladder
#   w10    345 latents  2,000 steps    ladder
#   w25    862 latents  2,000 steps    ladder
#   wtime  3,447        2,000 steps    control: training length alone
#   wdata  345          8,000 steps    control: data volume alone
#
# Stage 5 is skipped: these are guiding models, not candidates, and their
# standalone sample quality is not what they are for.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
AE=runs/il/ae/ae_enum_d16.pt
CK=runs/il/pretrain_ar_d16/pretrained.pt
[ -f "$AE" ] || { echo "MISSING $AE"; exit 1; }
[ -f runs/il/_fm_enum_fm_d16/last.pt ] || { echo "MISSING the strong model v1"; exit 1; }

echo ""; echo "=== AUTOGUIDANCE PHASE 0 START $(date) ==="

run () {  # tag  subset_n  steps  warmup
  echo ""; echo "=== $1  (subset ${2:-full}, $3 steps) ==="
  local sub=""
  [ "$2" != "0" ] && sub="--train-subset-n $2"
  timeout 7200 $PY scripts/il_finetune_fm.py --pretrained "$CK" \
    --latent-dim 16 --decoder ar --generative fm --sampler heun \
    --load-ae "$AE" $sub --train-subset-seed 0 \
    --fm-steps "$3" --fm-warmup "$4" --skip-stage5 \
    --tag "$1" 2>&1 | tee "runs/il/s25_$1.log"
  echo "--- exit $? at $(date) ---"
}

run ag_w05    172  2000 125
run ag_w10    345  2000 125
run ag_w25    862  2000 125
run ag_wtime    0  2000 125
run ag_wdata  345  8000 500

echo ""; echo "=== AUTOGUIDANCE PHASE 0 DONE $(date) ==="
