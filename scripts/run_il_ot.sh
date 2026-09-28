#!/usr/bin/env bash
# Does OT coupling make a richer source pay off?
#
# 6.15 found that swapping N(0,I) for a mixture fitted to the train cloud raised
# the PRIOR arm (0.491 -> 0.680) without moving `fm` at all, so gap-closed fell
# 86% -> 68%. The mixture took over the modelling and the endpoint did not improve.
#
# The diagnosis was the COUPLING, not the source. Under an independent coupling
# every step pairs a random x0 with an unrelated x1, so the field learns a map
# between arbitrary pairs however similar p0 and p1 are; moving p0 closer shrinks
# ||u_t|| while leaving its direction just as unpredictable, which is a
# worse-conditioned regression target rather than a better one.
#
# Minibatch OT pairs each x0 with a NEARBY x1 (exact Hungarian, fm/model.py:
# ot_pair -- it halves mean squared displacement at batch 128). That is what
# should turn proximity between the distributions into short, consistent
# displacements, and it is the one setting where a fitted source has a reason to
# help.
#
# This gives a 2x4 design against the four arms already on disk:
#     source in {gauss, full, gmm8, gmm32}  x  coupling in {independent, ot}
#
# Prediction worth recording before the numbers land: OT should help most where
# the source is richest, and the clearest signature is quality at LOW NFE, since
# straighter trajectories need fewer integration steps. Plausibility at 50 steps
# may barely move even when the trajectories have improved a lot.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
AE=runs/il/ae/ae_enum_d16.pt
CK=runs/il/pretrain_ar_d16/pretrained.pt
echo ""; echo "=== OT COUPLING RUN START $(date) ==="

run () {  # tag  source  k
  echo ""; echo "=== $1  (source=$2 ${3:+k=$3}, coupling=ot) ==="
  timeout 7200 $PY scripts/il_finetune_fm.py --pretrained "$CK" \
    --latent-dim 16 --decoder ar --generative fm --sampler heun \
    --load-ae "$AE" --source "$2" ${3:+--gmm-k $3} --coupling ot \
    --tag "$1" 2>&1 | tee "runs/il/s25_$1.log"
  echo "--- exit $? at $(date) ---"
}

run ot_gauss  gauss
run ot_full   full
run ot_gmm8   gmm   8
run ot_gmm32  gmm  32

echo ""; echo "=== OT COUPLING RUN DONE $(date) ==="
