#!/usr/bin/env bash
# Velocity-network ablation: does the architecture of the flow matter? Every
# setting matches sf_sig010 (SELFIES AE, full source, OT coupling, sigma_min 0.10,
# heun-50) except --fm-arch. Networks are parameter-matched to VelocityMLP's
# 2.58M in fm/nets.py. One seed; read against sf_sig010 and its run-to-run spread.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo ""; echo "=== ARCH RUN START $(date) ==="
for A in ffnn cnn unet; do
  T="sf_arch_$A"
  echo ""; echo "=== $T (selfies, full, ot, sigma_min=0.10, arch=$A) ==="
  timeout 7200 $PY scripts/il_finetune_fm.py \
    --pretrained runs/il/pretrain_sf_d16/pretrained.pt --tokenizer selfies --max-len 96 \
    --latent-dim 16 --decoder ar --generative fm --sampler heun \
    --load-ae runs/il/ae/ae_sf_d16.pt --source full --coupling ot --sigma-min 0.10 \
    --fm-arch "$A" --tag "$T" 2>&1 | tee "runs/il/s25_$T.log"
  echo "--- exit $? at $(date) ---"
done
echo ""; echo "=== ARCH RUN DONE $(date) ==="
