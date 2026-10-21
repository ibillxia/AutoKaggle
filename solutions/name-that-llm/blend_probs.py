# Blend kev holdout probabilities with TF-IDF probabilities on the same grouped split.
import json
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]

df = pd.read_csv("data/train.csv")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, ho_idx = next(gss.split(df, df["model"], groups=df["prompt_id"]))

# --- TF-IDF probabilities on holdout (same recipe as the sweep's best) ---
WORD = dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000)
CHAR = dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000)
Ftr, Fva = [], []
for cfg in (WORD, CHAR):
    vec = TfidfVectorizer(**cfg)
    Ftr.append(vec.fit_transform(df["text"].iloc[tr_idx]))
    Fva.append(vec.transform(df["text"].iloc[ho_idx]))
clf = LogisticRegression(C=30, max_iter=3000)
clf.fit(hstack(Ftr).tocsr(), df["model"].iloc[tr_idx])
classes = clf.classes_
p_tfidf = clf.predict_proba(hstack(Fva).tocsr())
P_tfidf = np.column_stack([p_tfidf[:, list(classes).index(m)] for m in MODELS])

# --- kev probabilities (dumped by eval_kev.py --dump_preds) ---
rows = [json.loads(l) for l in open("kev_preds.jsonl", encoding="utf-8")]
assert len(rows) == len(ho_idx)
P_kev = np.array([[r["probs"][m] for m in MODELS] for r in rows])
y_true = df["model"].iloc[ho_idx].values

def balacc(P):
    return balanced_accuracy_score(y_true, [MODELS[i] for i in P.argmax(1)])

print(f"tfidf alone : {balacc(P_tfidf):.4f}")
print(f"kev alone   : {balacc(P_kev):.4f}")

# geometric blend over alphas
Pt, Pk = np.clip(P_tfidf, 1e-9, 1), np.clip(P_kev, 1e-9, 1)
best = None
for a in np.arange(0, 1.01, 0.05):
    P = a * np.log(Pk) + (1 - a) * np.log(Pt)
    s = balacc(P)
    if best is None or s > best[1]:
        best = (a, s)
    print(f"  alpha(kev)={a:.2f}  bal_acc={s:.4f}")
print(f"best geometric blend: alpha={best[0]:.2f} -> {best[1]:.4f}")
