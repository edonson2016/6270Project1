#!/usr/bin/env bash
# Ionic-liquid latent flow matching -- overnight run.
# Ordered so the most informative arm completes first: if the night is cut
# short, the ChemBERTa/AR arm and its control are already on disk.
# Every stage writes incrementally; results.jsonl is appended per config.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
D=64
mkdir -p runs/il/results
echo "=== IL RUN START $(date) ==="

# Stage 0 must already be complete (scripts/il_embed.py).
for f in data/il_emb_ils.npy data/il_emb_ions.npy data/il_emb_moses.npy; do
  [ -f "$f" ] || { echo "MISSING $f -- Stage 0 incomplete, aborting"; exit 1; }
done
echo "Stage 0 artifacts present."

echo ""; echo "=== STAGE 1a: pretrain AR decoder (d=$D) ==="
timeout 12000 $PY scripts/il_pretrain.py --latent-dim $D --decoder ar --epochs 2 \
  --out runs/il/pretrain_ar 2>&1 | tee runs/il/stage1_ar.log
echo "--- exit $? at $(date) ---"

echo ""; echo "=== STAGE 2-5 (A): ChemBERTa latent, AR decoder ==="
timeout 7200 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_ar/pretrained.pt \
  --latent-dim $D --decoder ar --latent-mode chemberta --tag ar_chemberta 2>&1 | tee runs/il/s25_ar_chemberta.log
echo "--- exit $? at $(date) ---"

echo ""; echo "=== STAGE 2-5 (C): RANDOM-z control, AR decoder ==="
timeout 7200 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_ar/pretrained.pt \
  --latent-dim $D --decoder ar --latent-mode random --tag ar_random 2>&1 | tee runs/il/s25_ar_random.log
echo "--- exit $? at $(date) ---"

echo ""; echo "=== STAGE 1b: pretrain NAR decoder (d=$D) ==="
timeout 9000 $PY scripts/il_pretrain.py --latent-dim $D --decoder nar --epochs 2 \
  --out runs/il/pretrain_nar 2>&1 | tee runs/il/stage1_nar.log
echo "--- exit $? at $(date) ---"

echo ""; echo "=== STAGE 2-5 (B): ChemBERTa latent, NAR decoder ==="
timeout 7200 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_nar/pretrained.pt \
  --latent-dim $D --decoder nar --latent-mode chemberta --word-dropout 0 --tag nar_chemberta 2>&1 | tee runs/il/s25_nar_chemberta.log
echo "--- exit $? at $(date) ---"

echo ""; echo "=== ALL DONE $(date) ==="
cat runs/il/results/results.jsonl 2>/dev/null | wc -l | xargs echo "configs completed:"
