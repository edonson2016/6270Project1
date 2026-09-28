#!/usr/bin/env bash
# Does a richer SOURCE distribution help flow matching?
#
# The interpolant x_t = (1-t)x0 + t*x1 has conditional target u_t = x1 - x0,
# which does not depend on p0 at all -- so N(0,I) is a convention, not a
# requirement. A source closer to the data leaves the velocity field less
# transport to learn.
#
# The obvious trap is that it also leaves the field less to DO. Push the source
# onto the data and the flow learns the identity, and you are sampling from the
# source. So the number to watch is NOT plausibility -- it is the PRIOR arm,
# which decodes the source with no ODE at all, and the gap the flow closes on top
# of it. A source that lifts prior and fm by the same amount has bought nothing.
#
#   gauss   N(0, I)                      control, the current default
#   full    one full-covariance Gaussian  adds linear correlation between dims
#   gmm8    8-component diagonal mixture  adds multimodality
#   gmm32   32-component mixture          more modes, and a memorization risk
#
# §6.7 put the cloud's effective rank near 8 of 16, so `full` is not a trivial
# change. Watch novelty and uniqueness on the gmm arms: a source that memorizes
# the training cloud shows up as high plausibility with collapsed novelty.
#
# Everything but the source is held fixed: same autoencoder checkpoint, same
# frozen latent geometry, same splits, same 8,000 steps.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
AE=runs/il/ae/ae_enum_d16.pt
CK=runs/il/pretrain_ar_d16/pretrained.pt
echo ""; echo "=== SOURCE RUN START $(date) ==="

run () {  # tag  source  k
  echo ""; echo "=== $1  (source=$2 ${3:+k=$3}) ==="
  timeout 7200 $PY scripts/il_finetune_fm.py --pretrained "$CK" \
    --latent-dim 16 --decoder ar --generative fm --sampler heun \
    --load-ae "$AE" --source "$2" ${3:+--gmm-k $3} \
    --tag "$1" 2>&1 | tee "runs/il/s25_$1.log"
  echo "--- exit $? at $(date) ---"
}

run src_gauss  gauss
run src_full   full
run src_gmm8   gmm   8
run src_gmm32  gmm  32

echo ""; echo "=== SOURCE RUN DONE $(date) ==="
