# Realistic test: recover prompt groups on the holdout by clustering (no prompt_id at test time),
# then centre within each recovered cluster. Compare against oracle prompt_id centering.
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score

WORD = dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000)
CHAR = dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000)

df = pd.read_csv("data/train.csv")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, ho_idx = next(gss.split(df, df["model"], groups=df["prompt_id"]))

def center_by(M, groups):
    """subtract the group mean per row block, then L2-normalise rows"""
    M = M.tolil()
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        if len(rows) < 2:
            continue
        sub = M[rows].toarray()
        sub -= sub.mean(0, keepdims=True)
        M[rows] = sub
    M = M.tocsr()
    norms = np.sqrt(M.multiply(M).sum(1)).A.ravel()
    norms[norms == 0] = 1
    return M.multiply(1 / norms[:, None]).tocsr()

def build(groups_tr, groups_ho):
    """-> (train block, holdout block), both with the same column space"""
    tr_parts, ho_parts = [], []
    for cfg in (WORD, CHAR):
        vec = TfidfVectorizer(**cfg)
        A = vec.fit_transform(df["text"].iloc[tr_idx])
        B = vec.transform(df["text"].iloc[ho_idx])
        if groups_tr is not None:
            A = center_by(A, groups_tr)
        if groups_ho is not None:
            B = center_by(B, groups_ho)
        tr_parts.append(A); ho_parts.append(B)
    return hstack(tr_parts).tocsr(), hstack(ho_parts).tocsr()

ytr, yho = df["model"].iloc[tr_idx].values, df["model"].iloc[ho_idx].values
gtr = df["prompt_id"].iloc[tr_idx].values
gho_true = df["prompt_id"].iloc[ho_idx].values
ho_texts = df["text"].iloc[ho_idx].values

# recover clusters on the holdout alone (counts of prompts unknown -> try a few k)
wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), sublinear_tf=True, min_df=2, max_features=100000)
H = wv.fit_transform(ho_texts)

def score(A, B, tag):
    clf = LogisticRegression(C=30, max_iter=3000).fit(A, ytr)
    acc = balanced_accuracy_score(yho, clf.predict(B))
    print(f"{tag:44s} bal_acc={acc:.4f}", flush=True)

# --- variants ---
A, B = build(None, None)
score(A, B, "no centering")

A, B = build(gtr, gho_true)
score(A, B, "oracle prompt centering (train + holdout)")

for k in (18, 24, 30, 36, 45):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(H)
    A, B = build(gtr, km.labels_)
    score(A, B, f"train-oracle + holdout KMeans k={k}")
