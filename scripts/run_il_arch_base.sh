#!/usr/bin/env bash
# Second half of the velocity-network ablation: the SAME three architectures under
# BASELINE Stage-4 settings (gauss source, independent coupling, sigma_min 0), so
# architecture is crossed with the innovation rather than nested inside it.
#
# Reference cells already logged:
#   mlp + baseline = sf_base        mlp + recipe = sf_sig010
# Together with run_il_arch.sh this gives a 4 x 2 design and answers whether any
# architecture choice INTERACTS with OT coupling, or is simply additive.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo ""; echo "=== ARCH BASELINE RUN START $(date) ==="
for A in ffnn cnn unet; do
  T="sf_archb_$A"
  echo ""; echo "=== $T (selfies, gauss, independent, sigma_min=0, arch=$A) ==="
  timeout 10800 $PY scripts/il_finetune_fm.py \
    --pretrained runs/il/pretrain_sf_d16/pretrained.pt --tokenizer selfies --max-len 96 \
    --latent-dim 16 --decoder ar --generative fm --sampler heun \
    --load-ae runs/il/ae/ae_sf_d16.pt \
    --fm-arch "$A" --tag "$T" 2>&1 | tee "runs/il/s25_$T.log"
  echo "--- exit $? at $(date) ---"
done
echo ""; echo "=== ARCH BASELINE RUN DONE $(date) ==="
