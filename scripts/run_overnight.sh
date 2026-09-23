#!/usr/bin/env bash
# Overnight experiment run. Ordered so the more certain result lands first:
# if anything dies partway, Exp 2 is already complete on disk.
# Both scripts flush results.json after EVERY config, so partial results survive.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p runs/overnight

echo "=== START $(date) ==="

echo ""
echo "=== EXPERIMENT 2: conformer Pareto (latent vs handcrafted) ==="
timeout 14400 $PY scripts/exp2_conformer_pareto.py \
    --molecules ethanol,toluene,aspirin \
    --n 90000 --fm-steps 4000 --ae-epochs 15 --ae-width 256 \
    --widths 128,256 --depth 4 --n-samples 5000 --ode-steps 50 \
    --out runs/overnight/exp2 2>&1 | tee runs/overnight/exp2.log
echo "=== exp2 exit: $? at $(date) ==="

echo ""
echo "=== EXPERIMENT 1: MOSES latent-dim vs molecule size ==="
timeout 16200 $PY scripts/exp1_moses_scaling.py \
    --n-per-bin 15000 --latent-dims 16,32,64,128 \
    --epochs 12 --hidden 256 --emb-dim 96 \
    --beta 1e-4 --word-dropout 0.25 \
    --fm-steps 6000 --fm-width 384 --n-samples 2000 \
    --out runs/overnight/exp1 2>&1 | tee runs/overnight/exp1.log
echo "=== exp1 exit: $? at $(date) ==="

echo ""
echo "=== ANALYSIS ==="
$PY scripts/analyze_experiments.py 2>&1 | tee runs/overnight/analysis.log
echo "=== DONE $(date) ==="
