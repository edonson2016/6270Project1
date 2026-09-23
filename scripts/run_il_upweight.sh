#!/usr/bin/env bash
# RUN B -- is minority-mode coverage a density-weighting problem?
#
# §6.10 injected 150 ILThermo OOD pairs into a 3,447-pair training set and got a
# split verdict: the autoencoder absorbed them immediately (held-out OOD recon
# 0.9243 -> 0.9544, ceiling_ood 0.956 -> 1.000) while the flow barely moved
# (sw2_ood 0.538 -> 0.513 on matched references, inside +/-0.072 subset noise).
#
# The reading was that this is a DENSITY problem, not a representational one:
# 150 points against 3,597 is 4.2% of the mass, and a velocity field fitted by
# regression over the whole cloud will not spend capacity on a 4.2% mode. If
# that is right, replicating those same 150 rows should move sw2_ood; if the
# region is genuinely hard for the flow to reach, it should not.
#
# Everything is held fixed except the replication factor. The autoencoder is
# LOADED from §6.10's checkpoint, so the latent geometry is not merely similar
# but identical, and --upweight touches only the Stage 4 cloud. The corpus,
# splits, held-out 64 and decoder are all §6.10's.
#
#   1x   150/3,597   4.2% of density   <- control, reproduces §6.10
#   5x   750/4,197  17.9%
#  10x 1,500/4,947  30.3%
#
# The 1x arm is re-run rather than compared against §6.10's own numbers on
# purpose: §6.10's FM leg trained the autoencoder in-process, which consumed RNG
# draws before Stage 4 and left its flow on a different init stream than any
# --load-ae run. Re-running 1x here puts every arm on the identical stream, so
# the only difference between rows is the upweighting.
#
# Read sw2_ood against sw2_test_ion and plausibility: upweighting a shifted
# sub-population to ~30% of the density should cost the main distribution
# something, and the size of that cost is half the result.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
AE=runs/il/ae/ae_oodtrain_d16.pt
mkdir -p runs/il/results
[ -f "$AE" ] || { echo "MISSING $AE -- §6.10's autoencoder is required"; exit 1; }
echo ""; echo "=== UPWEIGHT RUN START $(date) ==="

for K in 1 5 10; do
  echo ""; echo "=== d=16 FM   upweight ${K}x (AE loaded from §6.10; flow cloud only) ==="
  timeout 7200 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_ar_d16/pretrained.pt \
    --latent-dim 16 --decoder ar --generative fm --sampler heun \
    --corpus il_pairs_v4 --n-original 4790 --ood-stem il_ood_pairs_v4 \
    --upweight $K --upweight-source ilthermo_ood_train --load-ae "$AE" \
    --tag up${K}_fm_d16 2>&1 | tee runs/il/s25_up${K}_fm_d16.log
  echo "--- exit $? at $(date) ---"

  echo ""; echo "=== d=16 DDPM upweight ${K}x (same autoencoder) ==="
  timeout 7200 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_ar_d16/pretrained.pt \
    --latent-dim 16 --decoder ar --generative ddpm --ddpm-T 1000 --ddpm-schedule linear \
    --sampler ancestral,ddim --clip-x0 3.0 \
    --corpus il_pairs_v4 --n-original 4790 --ood-stem il_ood_pairs_v4 \
    --upweight $K --upweight-source ilthermo_ood_train --load-ae "$AE" \
    --tag up${K}_ddpm_d16 2>&1 | tee runs/il/s25_up${K}_ddpm_d16.log
  echo "--- exit $? at $(date) ---"
done
echo ""; echo "=== UPWEIGHT RUN DONE $(date) ==="
