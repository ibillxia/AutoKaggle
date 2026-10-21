"""Verify the notebook's embed() implementation reproduces the stored training embeddings bit-for-bit.

Runs the same layer selection / mean pooling / length-sorted batching on 8 rows with 4-bit weights
(the configuration that produced emb_qwen4b_train.npz) and compares against the npz.
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

MODEL = "E:/Bill/AI/AutoKaggle/models/Qwen3.5-4B-Base"
LAYERS = [1, 3, 6, 9, 16, 24, 32]
MAXLEN = 320

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0}).eval()


@torch.no_grad()
def embed(texts, batch=8):
    idx_order = np.argsort([len(t.split()) for t in texts])
    out = np.zeros((len(texts), len(LAYERS) * model.config.text_config.hidden_size), dtype=np.float16)
    for i in range(0, len(idx_order), batch):
        idx = idx_order[i:i + batch]
        enc = tokenizer([texts[j] for j in idx], padding=True, truncation=True,
                        max_length=MAXLEN, return_tensors="pt").to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
        pooled = torch.cat([((hs[L].float() * mask).sum(1) / mask.sum(1).clamp_min(1)) for L in LAYERS], 1)
        out[idx] = pooled.to(torch.float16).cpu().numpy()
    assert not np.isnan(out).any()
    return out


tr = pd.read_csv("data/train.csv")
d = np.load("emb_qwen4b_train.npz", allow_pickle=True)
E_ref = np.concatenate([d[f"L{L}"] for L in LAYERS], 1)

rng = np.random.default_rng(0)
rows = rng.choice(len(tr), 8, replace=False)
mine = embed(tr["text"].iloc[rows].tolist())
ref = E_ref[rows]
cos = (mine.astype(np.float32) * ref.astype(np.float32)).sum(1) / (
    np.linalg.norm(mine.astype(np.float32), axis=1) * np.linalg.norm(ref.astype(np.float32), axis=1))
print("cosine per row:", np.round(cos, 5))
print("max abs diff:", float(np.abs(mine.astype(np.float32) - ref.astype(np.float32)).max()))
print("mean abs diff:", float(np.abs(mine.astype(np.float32) - ref.astype(np.float32)).mean()))
print("hidden_size from config:", model.config.text_config.hidden_size, "emb dim:", mine.shape[1])
