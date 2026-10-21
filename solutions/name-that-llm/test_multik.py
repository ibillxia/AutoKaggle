# Multi-k centering ensemble: centre with several recovered cluster counts and average probabilities.
# Robust against not knowing how many prompts the hidden test set contains.
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score

WORD = dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000)
CHAR = dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000)
MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]

df = pd.read_csv("data/train.csv")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, ho_idx = next(gss.split(df, df["model"], groups=df["prompt_id"]))
ytr, yho = df["model"].iloc[tr_idx].values, df["model"].iloc[ho_idx].values
gtr = df["prompt_id"].iloc[tr_idx].values

def center_by(M, groups):
    M = M.tolil()
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        if len(rows) < 2:
            continue
        sub = M[rows].toarray(); sub -= sub.mean(0, keepdims=True); M[rows] = sub
    M = M.tocsr()
    norms = np.sqrt(M.multiply(M).sum(1)).A.ravel(); norms[norms == 0] = 1
    return M.multiply(1 / norms[:, None]).tocsr()

vecs = []
for cfg in (WORD, CHAR):
    v = TfidfVectorizer(**cfg).fit(df["text"].iloc[tr_idx])
    vecs.append((v.transform(df["text"].iloc[tr_idx]), v.transform(df["text"].iloc[ho_idx])))
A0 = hstack([a for a, _ in vecs]).tocsr(); B0 = hstack([b for _, b in vecs]).tocsr()

wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), sublinear_tf=True, min_df=2, max_features=100000)
H = wv.fit_transform(df["text"].iloc[ho_idx])

def fit_predict(A, B):
    clf = LogisticRegression(C=30, max_iter=3000).fit(A, ytr)
    P = clf.predict_proba(B)
    order = [list(clf.classes_).index(m) for m in MODELS]   # probabilities are in sorted-class order
    return P[:, order]

def balacc(P):
    return balanced_accuracy_score(yho, np.array(MODELS)[P.argmax(1)])

probs = {}
Atr = hstack([center_by(a, gtr) for a, _ in vecs]).tocsr()
probs["oracle"] = fit_predict(Atr, B0 if False else hstack([center_by(b, df["prompt_id"].iloc[ho_idx].values) for _, b in vecs]).tocsr())
print(f"oracle centering        {balacc(probs['oracle']):.4f}", flush=True)

for k in (20, 24, 30, 36, 44):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(H)
    Bk = hstack([center_by(b, km.labels_) for _, b in vecs]).tocsr()
    probs[f"k{k}"] = fit_predict(Atr, Bk)
    print(f"KMeans k={k:<3}           {balacc(probs[f'k{k}']):.4f}", flush=True)

ks = [24, 30, 36]
Pmix = np.mean([probs[f"k{k}"] for k in ks], axis=0)
print(f"multi-k average {ks}  {balacc(Pmix):.4f}", flush=True)
np.savez("holdout_probs_tfidf.npz",
         **{k: v for k, v in probs.items()}, ymix=Pmix, y=yho,
         classes=np.array(MODELS))
