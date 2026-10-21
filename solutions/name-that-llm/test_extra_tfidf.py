# How much does the 24k extra corpus help the cheap TF-IDF model? (CPU, grouped holdout)
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
CFGS = [dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000),
        dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000),
        dict(analyzer="char_wb", ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_features=500000)]

tr = pd.read_csv("data/train.csv")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
a, b = next(gss.split(tr, tr["model"], groups=tr["prompt_id"]))
train, hold = tr.iloc[a].reset_index(drop=True), tr.iloc[b].reset_index(drop=True)
extra = pd.read_csv("extra/extra_short.csv")
print(f"train {len(train)} | holdout {len(hold)} | extra {len(extra)}")

def run(use_extra, cap, C=100):
    parts = [train[["text", "model"]]]
    if use_extra:
        ex = extra[["text", "model"]]
        if cap:
            ex = ex.groupby("model", group_keys=False).head(cap)
        parts.append(ex)
    fit = pd.concat(parts, ignore_index=True)
    Xfit, Xho = [], []
    for cfg in CFGS:
        v = TfidfVectorizer(**cfg).fit(fit["text"])
        Xfit.append(v.transform(fit["text"]))
        Xho.append(v.transform(hold["text"]))
    A, B = hstack(Xfit).tocsr(), hstack(Xho).tocsr()
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(A, fit["model"].values)
    pred = clf.predict(B)
    acc = balanced_accuracy_score(hold["model"].values, pred)
    print(f"extra={use_extra} cap={cap or '-'} C={C} rows={len(fit):5d}  bal_acc={acc:.4f}", flush=True)

run(False, 0)
run(True, 0)
run(True, 800)
run(True, 1500)
