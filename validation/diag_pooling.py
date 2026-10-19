"""Compare CLS vs mean-pooled features from the frozen encoder: how much label signal does each
carry, and how anisotropic are they? A linear probe tells us whether the encoder is the problem
or the pooling choice is.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_hf_nli2 import load_rows  # noqa: E402

ROOT = Path("E:/Bill/AI/AutoKaggle")
MODEL = str(ROOT / "models/xlm-roberta-large")
DATA = str(ROOT / "data/comp/contradictory-my-dear-watson/kev_train_pool.jsonl")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

from transformers import AutoTokenizer, XLMRobertaModel  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
enc_model = XLMRobertaModel.from_pretrained(MODEL)
enc_model.eval().cuda()

rows = load_rows(DATA, N)
y = np.array([r[2] for r in rows])
cls_feats, mean_feats = [], []
with torch.no_grad():
    for i in range(0, len(rows), 32):
        ch = rows[i:i + 32]
        e = tok([r[0] for r in ch], [r[1] for r in ch], padding=True, truncation=True,
                max_length=256, return_tensors="pt").to("cuda")
        h = enc_model(**e).last_hidden_state.float()
        m = e["attention_mask"].unsqueeze(-1).float()
        cls_feats.append(h[:, 0, :].cpu().numpy())
        mean_feats.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())

for name, X in (("CLS", np.concatenate(cls_feats)), ("MEAN", np.concatenate(mean_feats))):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    cos = Xn[:4] @ Xn[:4].T
    print("%s: |h|mean %.2f  cos(0,1) %.6f  cos(0,2) %.6f  feature std %.4f"
          % (name, np.linalg.norm(X, axis=1).mean(), cos[0, 1], cos[0, 2], X.std(0).mean()))

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    sc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
                         X, y, cv=3, n_jobs=3)
    print("   linear probe acc %.4f (+-%.4f)" % (sc.mean(), sc.std()))
