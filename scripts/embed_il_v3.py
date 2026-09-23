"""Embed the 933 CIR-resolved pairs and append them to the base ChemBERTa cache.

Lifted verbatim from the heredoc in run_il_extended.sh, which the restart killed
between build_il_merged.py and this step. Writes the _v3 embedding/SMILES files
that il_finetune_fm.py resolves from a `--corpus il_pairs_v3` stem.
"""
import numpy as np, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

DATA = Path("data"); MODEL = "seyonec/ChemBERTa-zinc-base-v1"
new = [s for s in (DATA/"il_smi_ils_new.txt").read_text().split("\n") if s]
old_E = np.load(DATA/"il_emb_ils.npy")
old_S = (DATA/"il_smi_ils.txt").read_text().split("\n")
assert len(old_E) == len(old_S), f"{len(old_E)} vs {len(old_S)}"

if new:
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = AutoModel.from_pretrained(MODEL).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(new), 64):
            b = tok(new[i:i+64], padding=True, truncation=True, max_length=128,
                    return_tensors="pt")
            h = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h*m).sum(1)/m.sum(1).clamp_min(1)).cpu())
            print(f"  {min(i+64,len(new))}/{len(new)}", flush=True)
    E = np.concatenate([old_E, torch.cat(out).numpy().astype(np.float32)])
else:
    E = old_E

np.save(DATA/"il_emb_ils_v3.npy", E)
(DATA/"il_smi_ils_v3.txt").write_text("\n".join(old_S + new))
print(f"embeddings {old_E.shape} + {len(new)} -> {E.shape}")
