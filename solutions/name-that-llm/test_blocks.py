# Final TF-IDF pipeline experiment: raw block + prompt-centred block, C grid, SVC vs LogReg.
# Holdout groups are recovered by KMeans (no prompt_id at test time).
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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

def blocks(groups_tr, groups_ho):
    """-> (train, holdout) with [raw_word raw_char | centred_word centred_char]"""
    A, B = [], []
    for cfg in (WORD, CHAR):
        vec = TfidfVectorizer(**cfg)
        Xtr = vec.fit_transform(df["text"].iloc[tr_idx])
        Xho = vec.transform(df["text"].iloc[ho_idx])
        A.append(Xtr); B.append(Xho)
        A.append(center_by(Xtr, groups_tr)); B.append(center_by(Xho, groups_ho))
    return hstack(A).tocsr(), hstack(B).tocsr()

wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), sublinear_tf=True, min_df=2, max_features=100000)
H = wv.fit_transform(df["text"].iloc[ho_idx])
k = int(round(len(ho_idx) / 18))     # ~18 responses per prompt in this dataset
km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(H)
print(f"recovered k={k} clusters for {len(ho_idx)} holdout rows", flush=True)

A, B = blocks(gtr, km.labels_)
for C in (3, 10, 30, 100, 300):
    clf = LogisticRegression(C=C, max_iter=3000).fit(A, ytr)
    print(f"raw+centred LogReg C={C:<4} bal_acc={balanced_accuracy_score(yho, clf.predict(B)):.4f}", flush=True)
for C in (0.3, 1, 3, 10):
    clf = LinearSVC(C=C).fit(A, ytr)
    print(f"raw+centred LinearSVC C={C:<4} bal_acc={balanced_accuracy_score(yho, clf.predict(B)):.4f}", flush=True)
