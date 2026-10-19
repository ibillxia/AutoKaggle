"""Is the supervision consistent with the input? Fit a plain linear probe on FROZEN XLM-R features
with a train/test split.

  test accuracy ~33%  -> the labels carry no information about the text: the data file is broken
  test accuracy 65-85% -> labels are fine, so the fault is in the fine-tuning loop

A 100%-on-train result is not evidence: a 1M-parameter head can fit 32 random labels. The split is
what makes this diagnostic meaningful.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "xlm-roberta-large"
CLASSES = ["entailment", "neutral", "contradiction"]
N = 2400


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


def feats(rows, bs=32):
    out = []
    for i in range(0, len(rows), bs):
        ch = rows[i:i + bs]
        enc = tok([r[0] for r in ch], [r[1] for r in ch], padding=True, truncation=True,
                  max_length=128, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model.roberta(**enc)
        out.append(o.last_hidden_state[:, 0, :].float().cpu().numpy())
    return np.concatenate(out, 0)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

for name in ["kev_train_pool.jsonl", "kev_train_ext.jsonl"]:
    path = ROOT / "data/comp/contradictory-my-dear-watson" / name
    if not path.exists():
        print(name, "missing", flush=True)
        continue
    rows = load(path, N)
    X = feats(rows)
    y = np.array([r[2] for r in rows])
    # split: first 75% train, last 25% test (rows are source-ordered, so also report a random split)
    ntr = int(len(rows) * 0.75)
    for tag, (tr, te) in {
        "ordered": (np.arange(ntr), np.arange(ntr, len(rows))),
        "random": None,
    }.items():
        if te is None:
            rng = np.random.RandomState(0)
            perm = rng.permutation(len(rows))
            tr, te = perm[:ntr], perm[ntr:]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        a_tr = clf.score(X[tr], y[tr])
        a_te = clf.score(X[te], y[te])
        print("%-22s %-8s n=%d  train acc %.4f  TEST acc %.4f" % (name, tag, len(rows), a_tr, a_te), flush=True)
        if tag == "random":
            print("   confusion:\n%s" % confusion_matrix(y[te], clf.predict(X[te])), flush=True)
    print("   class counts %s" % np.bincount(y, minlength=3).tolist(), flush=True)
