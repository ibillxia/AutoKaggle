# Qwen3-Embedding-0.6B embeddings -> linear classifier, grouped CV.
# Saves embeddings to npz so downstream blending doesn't need the GPU again.
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

MODEL_DIR = "E:/Bill/AI/AutoKaggle/models/Qwen3-Embedding-0.6B"
OUT = "emb_train.npz"
MAXLEN = 512

def last_token_pool(last_hidden, attention_mask):
    seq = attention_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.size(0)), seq]

@torch.no_grad()
def embed(texts, model, tok, device, batch=16):
    out = []
    t0 = time.time()
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=MAXLEN,
                  return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        e = last_token_pool(h, enc["attention_mask"])
        e = torch.nn.functional.normalize(e, dim=-1)
        out.append(e.float().cpu().numpy())
        if (i // batch) % 20 == 0:
            r = (i + batch) / len(texts)
            print(f"  {i + batch}/{len(texts)}  eta={(time.time()-t0)/max(r,1e-9)*(1-r):.0f}s", flush=True)
    return np.concatenate(out)

def main():
    df = pd.read_csv("data/train.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, padding_side="left")
    model = AutoModel.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(device).eval()
    print("model loaded", flush=True)

    E = embed(df["text"].tolist(), model, tok, device)
    np.savez(OUT, emb=E, y=df["model"].values, groups=df["prompt_id"].values)
    print(f"saved {OUT} {E.shape}", flush=True)

    y = df["model"].values
    groups = df["prompt_id"].values
    cv = GroupKFold(n_splits=5)
    y_pred = np.empty_like(y)
    for tr, va in cv.split(E, y, groups):
        clf = LogisticRegression(C=10, max_iter=3000)
        clf.fit(E[tr], y[tr])
        y_pred[va] = clf.predict(E[va])
    print(f"Qwen3-Emb-0.6B + LogReg C=10  bal_acc={balanced_accuracy_score(y, y_pred):.4f}", flush=True)

if __name__ == "__main__":
    main()
