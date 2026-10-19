"""Settle whether the supervision in the pool file is learnable at all.

Fits a plain logistic regression on FROZEN XLM-R features and reports both the fit and a held-out
score, with and without feature standardisation (XLM-R hidden states carry huge outlier dimensions,
which makes regularised probes underfit unless you normalise).

  fit ~100% on 128 rows, decent CV  -> data is fine, the fine-tuning loop is the fault
  fit ~40%                          -> the labels do not match the text
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "xlm-roberta-large"
DATA = ROOT / "data/comp/contradictory-my-dear-watson/kev_train_pool.jsonl"
CLASSES = ["entailment", "neutral", "contradiction"]


def load(path, n):
    rows = []
    for line in Path(path).open(encoding="utf-8"):
        rec = json.loads(line)
        body = rec["state"][len("Premise: "):]
        p, h = body.split("\n\nHypothesis: ", 1)
        lab = rec["questions"]["relation"]["label"]
        rows.append((p.strip(), h.strip(), CLASSES.index(lab) if isinstance(lab, str) else int(lab)))
        if len(rows) >= n:
            break
    return rows


tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3).to("cuda").eval()


def feats(rows, pool, bs=32):
    out = []
    for i in range(0, len(rows), bs):
        ch = rows[i:i + bs]
        enc = tok([r[0] for r in ch], [r[1] for r in ch], padding=True, truncation=True,
                  max_length=128, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model.roberta(**enc)
        h = o.last_hidden_state.float()
        if pool == "cls":
            v = h[:, 0, :]
        else:
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
        out.append(v.cpu().numpy())
    return np.concatenate(out, 0)


rows = load(DATA, 2400)
y = np.array([r[2] for r in rows])
print("rows %d  class counts %s" % (len(rows), np.bincount(y, minlength=3).tolist()), flush=True)

for pool in ("cls", "mean"):
    X = feats(rows, pool)
    for tag, est in {
        "raw": LogisticRegression(max_iter=3000, C=1.0),
        "standardised": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
    }.items():
        # (a) fit == eval on the same 128 rows: can a linear map even separate them?
        n = 128
        est.fit(X[:n], y[:n])
        fit_acc = est.score(X[:n], y[:n])
        # (b) honest generalisation: 75/25 split, shuffled
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(rows))
        tr, te = perm[:int(0.75 * len(rows))], perm[int(0.75 * len(rows)):]
        est.fit(X[tr], y[tr])
        cv = est.score(X[te], y[te])
        print("  pool=%-4s %-13s fit128 %.4f   holdout-acc %.4f" % (pool, tag, fit_acc, cv), flush=True)
