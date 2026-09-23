#!/usr/bin/env bash
# Follow-up to run_il_overnight.sh, CPU only.
#
#   Part 1  the missing 2x2 cell: NAR decoder x random-z control.
#           Reuses runs/il/pretrain_nar/pretrained.pt -- no new pretrain needed.
#   Part 2  the latent-dimension sweep, AR + ChemBERTa, d in {16,32,128}.
#           d=64 is already on disk as tag `ar_chemberta`, so it is not repeated.
#           Each d needs its OWN pretrain: MLP_down's output width and the
#           decoder's conditioning both depend on d, so a checkpoint from one d
#           cannot be loaded at another. That is the whole cost of this sweep.
#
# Ordered cheapest-first so a short night still finishes Part 1.
# results.jsonl is appended per config; analyze_il.py dedups by tag.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p runs/il/results
echo "=== IL SWEEP START $(date) ==="

# ---------------------------------------------------------------- Part 1
echo ""; echo "=== (D): RANDOM-z control, NAR decoder (d=64) ==="
if [ -f runs/il/pretrain_nar/pretrained.pt ]; then
  timeout 7200 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_nar/pretrained.pt \
    --latent-dim 64 --decoder nar --latent-mode random --word-dropout 0 \
    --tag nar_random 2>&1 | tee runs/il/s25_nar_random.log
  echo "--- exit $? at $(date) ---"
else
  echo "MISSING runs/il/pretrain_nar/pretrained.pt -- skipping (D)"
fi

# ---------------------------------------------------------------- Part 2
for D in 16 32 128; do
  echo ""; echo "=== STAGE 1: pretrain AR decoder (d=$D) ==="
  timeout 14400 $PY scripts/il_pretrain.py --latent-dim $D --decoder ar --epochs 2 \
    --out runs/il/pretrain_ar_d$D 2>&1 | tee runs/il/stage1_ar_d$D.log
  echo "--- exit $? at $(date) ---"

  echo ""; echo "=== STAGE 2-5: ChemBERTa latent, AR decoder (d=$D) ==="
  timeout 7200 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_ar_d$D/pretrained.pt \
    --latent-dim $D --decoder ar --latent-mode chemberta \
    --tag ar_chemberta_d$D 2>&1 | tee runs/il/s25_ar_chemberta_d$D.log
  echo "--- exit $? at $(date) ---"
done

echo ""; echo "=== SWEEP DONE $(date) ==="
$PY scripts/analyze_il.py 2>&1 | tail -40
