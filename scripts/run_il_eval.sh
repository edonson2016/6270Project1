#!/usr/bin/env bash
# Re-run the AR/ChemBERTa sweep under the split protocol: train on TRAIN,
# gate on VALID, score on TEST / TEST_ION / OOD. Pretrains already exist, so
# this is Stages 2-5 only -- about 30 min per point.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p runs/il/results
echo "=== IL SPLIT-EVAL START $(date) ==="
for D in 16 32 64 128; do
  CK=runs/il/pretrain_ar_d$D/pretrained.pt
  [ "$D" = "64" ] && CK=runs/il/pretrain_ar/pretrained.pt     # d=64 predates the _dNN naming
  if [ ! -f "$CK" ]; then echo "MISSING $CK -- skipping d=$D"; continue; fi
  echo ""; echo "=== STAGE 2-5 split protocol, AR/ChemBERTa d=$D ==="
  timeout 7200 $PY scripts/il_finetune_fm.py --pretrained "$CK" \
    --latent-dim $D --decoder ar --latent-mode chemberta \
    --tag ar_chemberta_d${D}_split 2>&1 | tee runs/il/s25_split_d$D.log
  echo "--- exit $? at $(date) ---"
done
echo ""; echo "=== SPLIT-EVAL DONE $(date) ==="
