#!/usr/bin/env bash
# sigma knee check + decodability-weighted Stage 4 sampling.
# Base recipe: selfies + full + OT + sigma 0.10 (best so far).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
B="--pretrained runs/il/pretrain_sf_d16/pretrained.pt --tokenizer selfies --max-len 96
   --latent-dim 16 --decoder ar --generative fm --sampler heun
   --load-ae runs/il/ae/ae_sf_d16.pt --source full --coupling ot"
echo ""; echo "=== DW RUN START $(date) ==="
run () { echo ""; echo "=== $1 ==="; shift
  timeout 7200 $PY scripts/il_finetune_fm.py $B "$@" 2>&1 | tee "runs/il/s25_$(echo "$@" | grep -o 'tag [^ ]*' | cut -d' ' -f2).log"
  echo "--- exit $? at $(date) ---"; }
run "sigma 0.20"            --sigma-min 0.20 --tag sf_sig020
run "decode-weight beta=1"  --sigma-min 0.10 --decode-weight 1 --tag sf_dw1
run "decode-weight beta=3"  --sigma-min 0.10 --decode-weight 3 --tag sf_dw3
echo ""; echo "=== DW RUN DONE $(date) ==="
