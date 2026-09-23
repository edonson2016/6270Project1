#!/usr/bin/env bash
# RUN A -- does a larger OBSERVED corpus reduce the flow's overfitting?
#
# §6.5 measured the flow sitting 2.3-3.1x closer to training latents than to
# held-out ones. §6.10 then showed that injecting 150 shifted OOD pairs moved
# the flow almost not at all (sw2_ood 0.538 -> 0.513, inside +/-0.072 noise)
# while the autoencoder absorbed them instantly -- which read as a density
# weighting problem, not a representational one.
#
# This is the other lever: +933 CIR-resolved ILThermo pairs, i.e. +27% of
# IN-DISTRIBUTION data, aimed straight at the overfitting gap. The number to
# judge it by is sw2_test_ion, NOT plausibility.
#
# The 150 OOD pairs are carried over so the OOD arm stays comparable to §6.10.
# The v5 split was rebuilt on top of v3 and verified to reproduce the v4 draw
# exactly: same 150 injected, same 64 held out, valid/test/test_ion unmoved
# (runs/il/build_v5.log). So against §6.10 the ONLY thing that changes is the
# 933 extra observed ILs; against §6.9 it is those plus the 150.
#
#   train 3,447 (§6.9)  ->  3,597 (§6.10, +150 OOD)  ->  4,530 (here, +933 CIR)
#
# Decoder curriculum (2a on the same 103k enumerated pairs) is unchanged, and
# one autoencoder is trained and shared by FM and DDPM so the comparison is exact.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
AE=runs/il/ae/ae_v3ood_d16.pt
mkdir -p runs/il/ae runs/il/results
echo ""; echo "=== V3+OOD RUN START $(date) ==="

echo ""; echo "=== d=16 FM  (train 4,530 = 3,447 zenodo + 933 CIR + 150 OOD; 64 OOD held out) ==="
timeout 21600 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_ar_d16/pretrained.pt \
  --latent-dim 16 --decoder ar --generative fm --sampler heun \
  --enum-pairs enum --enum-epochs 2 --corpus il_pairs_v5 --n-original 4790 \
  --ood-stem il_ood_pairs_v5 --save-ae "$AE" \
  --tag v3ood_fm_d16 2>&1 | tee runs/il/s25_v3ood_fm_d16.log
echo "--- exit $? at $(date) ---"

echo ""; echo "=== d=16 DDPM (same autoencoder) ==="
timeout 14400 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_ar_d16/pretrained.pt \
  --latent-dim 16 --decoder ar --generative ddpm --ddpm-T 1000 --ddpm-schedule linear \
  --sampler ancestral,ddim --clip-x0 3.0 --corpus il_pairs_v5 --n-original 4790 \
  --ood-stem il_ood_pairs_v5 --load-ae "$AE" \
  --tag v3ood_ddpm_d16 2>&1 | tee runs/il/s25_v3ood_ddpm_d16.log
echo "--- exit $? at $(date) ---"
echo ""; echo "=== V3+OOD RUN DONE $(date) ==="
