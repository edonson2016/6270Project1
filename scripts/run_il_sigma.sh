#!/usr/bin/env bash
# Stochastic interpolant: does noise around the path recover the diversity that OT
# coupling costs? OT lowered uniqueness 0.794 -> 0.758 (SMILES) by making the map
# more deterministic. sigma_min widens the region of x-space the field sees without
# changing the conditional target u_t = x1 - x0, so it should trade back.
# Read UNIQUENESS and novel_cation against plausibility.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo ""; echo "=== SIGMA RUN START $(date) ==="
for S in 0.02 0.05 0.10; do
  T="sf_sig${S/./}"
  echo ""; echo "=== $T (selfies, full, ot, sigma_min=$S) ==="
  timeout 7200 $PY scripts/il_finetune_fm.py \
    --pretrained runs/il/pretrain_sf_d16/pretrained.pt --tokenizer selfies --max-len 96 \
    --latent-dim 16 --decoder ar --generative fm --sampler heun \
    --load-ae runs/il/ae/ae_sf_d16.pt --source full --coupling ot --sigma-min "$S" \
    --tag "$T" 2>&1 | tee "runs/il/s25_$T.log"
  echo "--- exit $? at $(date) ---"
done
echo ""; echo "=== SIGMA RUN DONE $(date) ==="
