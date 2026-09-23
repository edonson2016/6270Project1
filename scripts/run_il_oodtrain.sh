#!/usr/bin/env bash
# Can the flow cover the ILThermo region if shown part of it?
#
# sw2_ood has been flat at 0.46-0.55 across every model, sampler and d, while
# §6.7 showed those latents are NOT off-manifold -- just shifted 0.30-0.58 SD
# within it. That leaves open whether the flow CAN represent the shifted region
# or simply never saw it. Here 150 of the 227 OOD pairs go into TRAIN and 64 are
# held out (13 were blocked by the test_ion cation guard).
#
# Two readings:
#   sw2 on the held-out 64  -> does partial exposure generalize across the shift?
#   test / test_ion         -> what does injecting a shifted sub-population COST?
#
# Base is the ORIGINAL corpus, not the CIR-extended one, so this isolates the
# OOD injection. Decoder curriculum (2a on the same 103k) is unchanged, and
# valid/test/test_ion are the identical molecules as §6.9.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
AE=runs/il/ae/ae_oodtrain_d16.pt
mkdir -p runs/il/ae
echo "=== OOD-IN-TRAIN RUN START $(date) ==="

echo ""; echo "=== d=16 FM  (train 3447 -> 3597, 64 OOD held out) ==="
timeout 21600 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_ar_d16/pretrained.pt \
  --latent-dim 16 --decoder ar --generative fm --sampler heun \
  --enum-pairs enum --enum-epochs 2 --corpus il_pairs_v4 --n-original 4790 \
  --ood-stem il_ood_pairs_v4 --save-ae "$AE" \
  --tag oodtr_fm_d16 2>&1 | tee runs/il/s25_oodtr_fm_d16.log
echo "--- exit $? at $(date) ---"

echo ""; echo "=== d=16 DDPM (same autoencoder) ==="
timeout 14400 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_ar_d16/pretrained.pt \
  --latent-dim 16 --decoder ar --generative ddpm --ddpm-T 1000 --ddpm-schedule linear \
  --sampler ancestral,ddim --clip-x0 3.0 --corpus il_pairs_v4 --n-original 4790 \
  --ood-stem il_ood_pairs_v4 --load-ae "$AE" \
  --tag oodtr_ddpm_d16 2>&1 | tee runs/il/s25_oodtr_ddpm_d16.log
echo "--- exit $? at $(date) ---"
echo ""; echo "=== OOD-IN-TRAIN RUN DONE $(date) ==="
