#!/usr/bin/env bash
# Does raising the decoder ceiling change the FM-vs-DDPM verdict?
#
# At d=16 flow matching already closes 98.2% of the prior->ceiling gap, so the
# comparison at the best operating point has ~2% of headroom and cannot
# discriminate. This raises the ceiling by giving the decoder 30x more
# supervision, then re-runs both models against it.
#
# Curriculum (docs §2): phases 2a and 2b are the same operation on different
# data. 2a builds decoder capability on ~103k enumerated TRAIN-ion pairs; 2b
# re-seats the latent on the 3,447 observed pairs, because Stage 3 freezes
# whatever geometry the autoencoder ends on and the flow must model the REAL
# distribution, not the uniform product of ions.
#
# The autoencoder is identical for FM and DDPM, so it is trained once per d and
# reused via --save-ae/--load-ae. That also makes the comparison exact: both
# models see literally the same decoder and the same z cloud.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p runs/il/results runs/il/ae

until [ -f data/il_emb_enum.npy ]; do sleep 60; done
sleep 20                                    # let the writer finish
echo "=== ENUM RUN START $(date) ==="
$PY - <<'PYX'
import numpy as np
E=np.load("data/il_emb_enum.npy"); S=open("data/il_smi_enum.txt").read().split("\n")
print(f"enum corpus ready: {E.shape}, {len(S)} smiles")
assert len(E)==len(S)
PYX

for D in 16 64; do
  CK=runs/il/pretrain_ar_d$D/pretrained.pt
  [ "$D" = "64" ] && CK=runs/il/pretrain_ar/pretrained.pt
  AE=runs/il/ae/ae_enum_d$D.pt
  [ -f "$CK" ] || { echo "MISSING $CK -- skipping d=$D"; continue; }

  echo ""; echo "=== d=$D  FM  (phases 2a+2b, then flow) ==="
  timeout 21600 $PY scripts/il_finetune_fm.py --pretrained "$CK" --latent-dim $D \
    --decoder ar --generative fm --sampler heun \
    --enum-pairs enum --enum-epochs 2 --save-ae "$AE" \
    --tag enum_fm_d$D 2>&1 | tee runs/il/s25_enum_fm_d$D.log
  echo "--- exit $? at $(date) ---"

  echo ""; echo "=== d=$D  DDPM (reusing the same autoencoder) ==="
  timeout 14400 $PY scripts/il_finetune_fm.py --pretrained "$CK" --latent-dim $D \
    --decoder ar --generative ddpm --ddpm-T 1000 --ddpm-schedule linear \
    --sampler ancestral,ddim --clip-x0 3.0 --load-ae "$AE" \
    --tag enum_ddpm_d$D 2>&1 | tee runs/il/s25_enum_ddpm_d$D.log
  echo "--- exit $? at $(date) ---"
done

echo ""; echo "=== ENUM RUN DONE $(date) ==="
$PY scripts/analyze_il.py 2>&1 | tail -20
