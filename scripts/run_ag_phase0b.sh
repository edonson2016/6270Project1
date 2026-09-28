#!/usr/bin/env bash
# AUTOGUIDANCE PHASE 0b -- ladder over TRAINING LENGTH only, at full data.
#
# Phase 0 tested degradation by subsetting and the gate rejected it: 4 of 5
# candidates had E<e1,D> > 0 at 6-34 SE, meaning guidance strictly increases
# error. The per-t breakdown showed why -- a flow trained on 172-862 latents
# MEMORIZES, so its error points at its own subset rather than being a scaled
# copy of v1's, and the two error fields come out near-orthogonal.
#
# The controls did isolate one thing: the SIGN tracked training length, not data
# volume. At fixed data (345) the numerator flipped -0.207 <- +0.255 when steps
# went 2,000 -> 8,000. So this ladder holds the data fixed at all 3,447 latents
# -- identical distribution, identical cloud, identical Standardizer as v1 --
# and degrades ONLY by stopping early. That is the degradation most likely to
# leave e0 a scaled version of e1, since an undertrained model on the full
# distribution undersmooths rather than memorizing.
#
# v1 itself is the 8,000-step end of this same ladder, so every rung differs
# from it in exactly one scalar.
#
# Warmup scales with steps (500 * steps/8000); leaving it at 500 would put a
# 125-step run entirely inside its warmup and change the schedule shape.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
AE=runs/il/ae/ae_enum_d16.pt
CK=runs/il/pretrain_ar_d16/pretrained.pt

echo ""; echo "=== AUTOGUIDANCE PHASE 0b START $(date) ==="

run () {  # tag  steps  warmup
  echo ""; echo "=== $1  (full 3,447 latents, $2 steps) ==="
  timeout 3600 $PY scripts/il_finetune_fm.py --pretrained "$CK" \
    --latent-dim 16 --decoder ar --generative fm --sampler heun \
    --load-ae "$AE" --fm-steps "$2" --fm-warmup "$3" --skip-stage5 \
    --tag "$1" 2>&1 | tee "runs/il/s25_$1.log"
  echo "--- exit $? at $(date) ---"
}

run ag_t0125   125   8
run ag_t0250   250  16
run ag_t0500   500  31
run ag_t1000  1000  63
run ag_t4000  4000 250

echo ""; echo "=== AUTOGUIDANCE PHASE 0b DONE $(date) ==="
