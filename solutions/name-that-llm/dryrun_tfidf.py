# Local dry-run of the TF-IDF notebook logic: train on 120 prompts, treat the 30-prompt holdout as the
# hidden test set (prompt_id hidden, clusters recovered by KMeans), then score balanced accuracy.
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit

WORD = dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000)
CHAR = dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000)

def centre(M, groups):
    M = M.tolil()
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        if len(rows) < 2:
            continue
        sub = M[rows].toarray(); sub -= sub.mean(0, keepdims=True); M[rows] = sub
    M = M.tocsr()
    n = np.sqrt(M.multiply(M).sum(1)).A.ravel(); n[n == 0] = 1
    return M.multiply(1 / n[:, None]).tocsr()

df = pd.read_csv("data/train.csv")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, ho_idx = next(gss.split(df, df["model"], groups=df["prompt_id"]))
train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[ho_idx].reset_index(drop=True)
truth = test["model"].values
print(f"train {len(train)} rows / {train.prompt_id.nunique()} prompts | test {len(test)} rows")

Xtr, Xte = [], []
for cfg in (WORD, CHAR):
    v = TfidfVectorizer(**cfg).fit(train["text"])
    Xtr.append(v.transform(train["text"])); Xte.append(v.transform(test["text"]))
A = hstack([centre(x, train["prompt_id"].values) for x in Xtr]).tocsr()
clf = LogisticRegression(C=30, max_iter=3000).fit(A, train["model"].values)
classes = list(clf.classes_)

ratio = len(train) / train["prompt_id"].nunique()
k_base = max(2, int(round(len(test) / ratio)))
print(f"rows/prompt={ratio:.1f} -> k_base={k_base} (true={test.prompt_id.nunique()})")

wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), sublinear_tf=True, min_df=2, max_features=100000)
H = wv.fit_transform(test["text"])
probs = []
for k in sorted({max(2, int(round(k_base * f))) for f in (0.8, 1.0, 1.2)}):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(H)
    B = hstack([centre(x, lab) for x in Xte]).tocsr()
    p = clf.predict_proba(B)
    probs.append(p)
    acc = balanced_accuracy_score(truth, [classes[i] for i in p.argmax(1)])
    print(f"  k={k:<3} bal_acc={acc:.4f}")
P = np.mean(probs, axis=0)
pred = [classes[i] for i in P.argmax(1)]
print(f"multi-k average bal_acc = {balanced_accuracy_score(truth, pred):.4f}")
