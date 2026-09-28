#!/usr/bin/env bash
# Seed-replicate the §6.16 headline: full+OT vs gauss+independent.
# Splits are fixed by --split-seed 0, so replicates differ only in init/sampling RNG.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
AE=runs/il/ae/ae_enum_d16.pt
CK=runs/il/pretrain_ar_d16/pretrained.pt
echo ""; echo "=== SEED RUN START $(date) ==="
for S in 1 2; do
  for CFGN in "gauss independent" "full ot"; do
    set -- $CFGN
    T="seed${S}_$1_$2"
    echo ""; echo "=== $T ==="
    timeout 7200 $PY scripts/il_finetune_fm.py --pretrained "$CK" \
      --latent-dim 16 --decoder ar --generative fm --sampler heun \
      --load-ae "$AE" --source "$1" --coupling "$2" --seed "$S" \
      --tag "$T" 2>&1 | tee "runs/il/s25_$T.log"
    echo "--- exit $? at $(date) ---"
  done
done
echo ""; echo "=== SEED RUN DONE $(date) ==="
