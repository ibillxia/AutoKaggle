# Does per-prompt centering of TF-IDF features help? (removes topic, keeps style)
# Train-centering uses known prompt_id; holdout-centering simulates recovering prompt clusters at test time.
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score

WORD = dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000)
CHAR = dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000)

df = pd.read_csv("data/train.csv")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, ho_idx = next(gss.split(df, df["model"], groups=df["prompt_id"]))

def build(center):
    Ftr, Fho = [], []
    for cfg in (WORD, CHAR):
        vec = TfidfVectorizer(**cfg)
        A = vec.fit_transform(df["text"].iloc[tr_idx])
        B = vec.transform(df["text"].iloc[ho_idx])
        if center:
            for M, groups, idx in ((A, df["prompt_id"].iloc[tr_idx].values, None),
                                   (B, df["prompt_id"].iloc[ho_idx].values, None)):
                M = M.tolil()
                for g in np.unique(groups):
                    rows = np.where(groups == g)[0]
                    if len(rows) < 2:
                        continue
                    sub = M[rows].toarray()
                    sub -= sub.mean(0, keepdims=True)
                    M[rows] = sub
                M = csr_matrix(M)
                # renormalise so the scale matches the uncentered features
                norms = np.sqrt(M.multiply(M).sum(1)).A.ravel()
                norms[norms == 0] = 1
                M = csr_matrix(M.multiply(1 / norms[:, None]).tocsr())
                if idx is None and M.shape[0] == A.shape[0]:
                    A = M
                else:
                    B = M
        Ftr.append(A); Fho.append(B)
    return hstack(Ftr).tocsr(), hstack(Fho).tocsr()

ytr, yho = df["model"].iloc[tr_idx].values, df["model"].iloc[ho_idx].values
for center in (False, True):
    A, B = build(center)
    clf = LogisticRegression(C=30, max_iter=3000).fit(A, ytr)
    acc = balanced_accuracy_score(yho, clf.predict(B))
    print(f"center={center}: holdout bal_acc={acc:.4f}", flush=True)
