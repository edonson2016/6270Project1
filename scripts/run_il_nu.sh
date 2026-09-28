#!/usr/bin/env bash
# Heavy-tailed source. Measured motivation: high-melting ILs sit further from the
# train-cloud centroid (radius 3.66 -> 4.52 across melting-point bands) and are
# generated worst (plausibility 0.976 -> 0.908), even though the autoencoder
# reconstructs them BETTER (exact 0.377 -> 0.468). So the failure is the flow's, at
# the periphery -- exactly where a Gaussian source puts least mass.
# Student-t keeps `full`'s mean and covariance and only adds tail weight.
# Read: per-band plausibility spread, and sw2_test/test_ion.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
B="--pretrained runs/il/pretrain_sf_d16/pretrained.pt --tokenizer selfies --max-len 96
   --latent-dim 16 --decoder ar --generative fm --sampler heun
   --load-ae runs/il/ae/ae_sf_d16.pt --coupling ot --sigma-min 0.10"
echo ""; echo "=== NU RUN START $(date) ==="
for NU in 3 5 10; do
  echo ""; echo "=== sf_nu$NU (studentt, nu=$NU) ==="
  timeout 7200 $PY scripts/il_finetune_fm.py $B --source studentt --nu $NU --tag "sf_nu$NU" 2>&1 \
    | tee "runs/il/s25_sf_nu$NU.log"
  echo "--- exit $? at $(date) ---"
done
echo ""; echo "=== NU RUN DONE $(date) ==="
