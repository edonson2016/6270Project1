#!/usr/bin/env bash
# Does a larger OBSERVED corpus reduce the flow's overfitting?
#
# §6.5 measured the flow sitting 2.3-3.1x closer to training latents than to
# held-out ones, and §6.7 showed the manifold's effective rank is flat with
# sample count -- so more real ILs should improve GENERALIZATION, not richness.
# The number to judge this by is sw2_test_ion, not plausibility.
#
# Everything except the flow's training set is held fixed against §6.9:
# identical decoder curriculum (the same 103k enumerated corpus at 2a), and
# valid / test / test_ion are literally the same molecules (see
# build_il_merged.py and make_splits_extended).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
WAIT=${1:-3600}

echo "=== EXTENDED RUN: harvesting for ${WAIT}s more, then training $(date) ==="
sleep "$WAIT"
for pid in $(pgrep -f "[r]esolve_ilthermo"); do kill "$pid" 2>/dev/null && echo "stopped resolver $pid"; done
sleep 3

$PY scripts/build_il_merged.py 2>&1 | tee runs/il/merge.log

echo ""; echo "=== embedding the new pairs and appending ==="
$PY - <<'PYX'
import numpy as np, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
DATA=Path("data"); MODEL="seyonec/ChemBERTa-zinc-base-v1"
new=[s for s in (DATA/"il_smi_ils_new.txt").read_text().split("\n") if s]
old_E=np.load(DATA/"il_emb_ils.npy"); old_S=(DATA/"il_smi_ils.txt").read_text().split("\n")
if new:
    tok=AutoTokenizer.from_pretrained(MODEL); enc=AutoModel.from_pretrained(MODEL).eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(new),64):
            b=tok(new[i:i+64],padding=True,truncation=True,max_length=128,return_tensors="pt")
            h=enc(**b).last_hidden_state; m=b["attention_mask"].unsqueeze(-1).float()
            out.append(((h*m).sum(1)/m.sum(1).clamp_min(1)).cpu())
    E=np.concatenate([old_E, torch.cat(out).numpy().astype(np.float32)])
else:
    E=old_E
np.save(DATA/"il_emb_ils_v3.npy", E)
(DATA/"il_smi_ils_v3.txt").write_text("\n".join(old_S+new))
print(f"embeddings {old_E.shape} + {len(new)} -> {E.shape}")
PYX

for MODEL_KIND in fm ddpm; do
  AE=runs/il/ae/ae_ext_d16.pt
  if [ "$MODEL_KIND" = "fm" ]; then
    echo ""; echo "=== EXTENDED d=16 FM (2a identical, 2b on the larger train split) ==="
    timeout 21600 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_ar_d16/pretrained.pt \
      --latent-dim 16 --decoder ar --generative fm --sampler heun \
      --enum-pairs enum --enum-epochs 2 --corpus il_pairs_v3 --n-original 4790 \
      --save-ae "$AE" --tag ext_fm_d16 2>&1 | tee runs/il/s25_ext_fm_d16.log
  else
    echo ""; echo "=== EXTENDED d=16 DDPM (same autoencoder) ==="
    timeout 14400 $PY scripts/il_finetune_fm.py --pretrained runs/il/pretrain_ar_d16/pretrained.pt \
      --latent-dim 16 --decoder ar --generative ddpm --ddpm-T 1000 --ddpm-schedule linear \
      --sampler ancestral,ddim --clip-x0 3.0 --corpus il_pairs_v3 --n-original 4790 \
      --load-ae "$AE" --tag ext_ddpm_d16 2>&1 | tee runs/il/s25_ext_ddpm_d16.log
  fi
  echo "--- exit $? at $(date) ---"
done
echo ""; echo "=== EXTENDED RUN DONE $(date) ==="
