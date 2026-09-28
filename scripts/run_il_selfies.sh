#!/usr/bin/env bash
# SELFIES decoder target. Hypothesis: 74% of the decoder ceiling's 9.7pp loss is
# SMILES PARSE failure (7.4pp), and SELFIES strings decode to a valid molecule by
# construction, so that term goes to zero.
#
# Risk: it converts a DETECTABLE failure into an undetectable one. An unparseable
# SMILES is rejected; a wrong SELFIES is a valid molecule that may silently be one
# fragment or charge-unbalanced. Matched-rate corruption favours SELFIES anyway
# (charge balance 0.474 vs 0.097 at 5% token corruption), but random substitution
# models a trained decoder's errors poorly, so the ceiling arm is the real test.
#
# Stage 1 must be redone: the output vocabulary changes from 767 BPE tokens to 102
# SELFIES tokens, so the embedding and output layers change shape. The ENCODER side
# is untouched -- ChemBERTa still reads SMILES and the cached embeddings are reused.
#
# max_len 96, not 80: SELFIES strings run slightly longer (mean 36.7 vs 32) and 80
# would truncate 1.95% of the corpus against 0.96% at 96.
#
# Read: ceiling_test parses (was 0.9258) and net_charge_zero (was 0.9165).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -f data/selfies_vocab.json ] || { echo "MISSING data/selfies_vocab.json"; exit 1; }
echo ""; echo "=== SELFIES RUN START $(date) ==="

echo ""; echo "=== STAGE 1: pretrain AR decoder on SELFIES (d=16) ==="
timeout 21600 $PY scripts/il_pretrain.py --latent-dim 16 --decoder ar --epochs 2 \
  --tokenizer selfies --max-len 96 --out runs/il/pretrain_sf_d16 \
  2>&1 | tee runs/il/stage1_sf_d16.log
echo "--- exit $? at $(date) ---"

for CFG in "gauss independent sf_base" "full ot sf_ot_full"; do
  set -- $CFG
  echo ""; echo "=== STAGE 2-5 selfies  source=$1 coupling=$2 -> $3 ==="
  AE=runs/il/ae/ae_sf_d16.pt
  EXTRA="--save-ae $AE --enum-pairs enum --enum-epochs 2"
  [ -f "$AE" ] && EXTRA="--load-ae $AE"
  timeout 21600 $PY scripts/il_finetune_fm.py \
    --pretrained runs/il/pretrain_sf_d16/pretrained.pt \
    --tokenizer selfies --max-len 96 \
    --latent-dim 16 --decoder ar --generative fm --sampler heun \
    --source "$1" --coupling "$2" $EXTRA \
    --tag "$3" 2>&1 | tee "runs/il/s25_$3.log"
  echo "--- exit $? at $(date) ---"
done
echo ""; echo "=== SELFIES RUN DONE $(date) ==="
