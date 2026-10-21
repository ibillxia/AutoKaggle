# Blend 4B layer embeddings with the TF-IDF + stylometry model on the same grouped holdout.
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, ".")
from test_style_features import style_features  # reuse the feature extractor

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
truth = hold["model"].values

Xtr, Xho = [], []
for cfg in CFGS:
    v = TfidfVectorizer(**cfg).fit(train["text"])
    Xtr.append(v.transform(train["text"])); Xho.append(v.transform(hold["text"]))
sc = StandardScaler().fit(style_features(train["text"]))
A = hstack(Xtr + [csr_matrix(sc.transform(style_features(train["text"])) * 0.3)]).tocsr()
B = hstack(Xho + [csr_matrix(sc.transform(style_features(hold["text"])) * 0.3)]).tocsr()

clf = LogisticRegression(C=100, max_iter=3000).fit(A, train["model"].values)
order = [list(clf.classes_).index(m) for m in MODELS]
P_tf = clf.predict_proba(B)[:, order]

z = np.load("emb_holdout_probs.npz", allow_pickle=True)
P_emb = z["probs"]
assert (z["y"] == truth).all()

def acc(P):
    return balanced_accuracy_score(truth, [MODELS[i] for i in np.argmax(P, 1)])

print(f"tfidf+style {acc(P_tf):.4f}\nemb(4B)     {acc(P_emb):.4f}")
Lt, Le = np.log(np.clip(P_tf, 1e-9, 1)), np.log(np.clip(P_emb, 1e-9, 1))
best = None
for w in np.arange(0, 1.01, 0.1):
    s = acc((1 - w) * Lt + w * Le)
    print(f"  w_emb={w:.1f} {s:.4f}")
    if best is None or s > best[1]:
        best = (round(w, 2), s)
print("best:", best)
np.savez("holdout_probs_tfidf_style.npz", probs=P_tf, y=truth, classes=np.array(MODELS))
