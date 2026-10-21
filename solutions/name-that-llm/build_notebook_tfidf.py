# Build the self-contained TF-IDF notebook (no external datasets): centred char+word TF-IDF + LogReg,
# multi-k prompt-cluster centering at test time.
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "ntl_tfidf.ipynb"


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = [
    md("""# Name That LLM — prompt-centred TF-IDF
Surface style (char n-grams, markdown/punctuation habits) carries most of the usable signal.
Centring features inside each prompt removes the topic component; at test time prompts are
recovered with KMeans, and predictions are averaged over three plausible cluster counts."""),

    code("""\
import glob, os
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def find(name):
    for pat in (f"/kaggle/input/competitions/name-that-llm/{name}",
                f"/kaggle/input/name-that-llm/{name}"):
        if os.path.exists(pat):
            return pat
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, name
    return hits[0]

train = pd.read_csv(find("train.csv"))
test = pd.read_csv(find("test.csv"))
print(train.shape, test.shape)
print(train["model"].value_counts().to_string())"""),

    code("""\
WORD = dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000)
CHAR = dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000)

def centre(M, groups):
    M = M.tolil()
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        if len(rows) < 2:
            continue
        sub = M[rows].toarray()
        sub -= sub.mean(0, keepdims=True)
        M[rows] = sub
    M = M.tocsr()
    n = np.sqrt(M.multiply(M).sum(1)).A.ravel()
    n[n == 0] = 1
    return M.multiply(1 / n[:, None]).tocsr()

Xtr, Xte = [], []
for cfg in (WORD, CHAR):
    v = TfidfVectorizer(**cfg).fit(train["text"])
    Xtr.append(v.transform(train["text"]))
    Xte.append(v.transform(test["text"]))
print("features:", [x.shape for x in Xtr])

A = hstack([centre(x, train["prompt_id"].values) for x in Xtr]).tocsr()
clf = LogisticRegression(C=30, max_iter=3000).fit(A, train["model"].values)
classes = list(clf.classes_)
print("trained on", A.shape, "classes", len(classes))"""),

    code("""\
# how many prompts does the test set hold? keep the train rows-per-prompt ratio
ratio = len(train) / train["prompt_id"].nunique()
k_base = max(2, int(round(len(test) / ratio)))
print(f"train rows/prompt={ratio:.1f} -> inferred test prompts ~{k_base}")

wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), sublinear_tf=True, min_df=2, max_features=100000)
H = wv.fit_transform(test["text"])

probs = []
if len(test) < 6:                      # interactive session: single example row, nothing to cluster
    B = hstack(Xte).tocsr()
    probs.append(clf.predict_proba(B))
    print("tiny preview test set -> no centring")
else:
    for k in sorted({min(len(test) - 1, max(2, int(round(k_base * f)))) for f in (0.8, 1.0, 1.2)}):
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(H)
        B = hstack([centre(x, lab) for x in Xte]).tocsr()
        probs.append(clf.predict_proba(B))
        print(f"  k={k} done")
P = np.mean(probs, axis=0)
pred = [classes[i] for i in P.argmax(1)]
print(pd.Series(pred).value_counts().to_string())"""),

    code("""\
sub = pd.DataFrame({"id": test["id"], "model": pred})
assert sub["id"].is_unique and sub["model"].notna().all() and len(sub) == len(test)
sub.to_parquet("submission.parquet", index=False)
print(sub.head())
print("wrote submission.parquet", sub.shape)"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}

if __name__ == "__main__":
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": "ibillxia/ntl-tfidf", "title": "ntl tfidf", "code_file": OUT.name,
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": False, "enable_internet": False,
            "dataset_sources": [], "competition_sources": ["name-that-llm"], "kernel_sources": []}
    (HERE / "kernel-metadata-tfidf.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", OUT)
