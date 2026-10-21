# Extra squeeze: average centred predictions over {C} x {char n-gram ranges} for stability.
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
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
WORD = dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000)
C25 = dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000)
C36 = dict(analyzer="char_wb", ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_features=500000)

df = pd.read_csv("data/train.csv")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, ho_idx = next(gss.split(df, df["model"], groups=df["prompt_id"]))
train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[ho_idx].reset_index(drop=True)
truth = test["model"].values

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

stacks = {}
for tag, cfg_list in (("c25", (WORD, C25)), ("c36", (WORD, C36)), ("both", (WORD, C25, C36))):
    Xtr, Xte = [], []
    for cfg in cfg_list:
        v = TfidfVectorizer(**cfg).fit(train["text"])
        Xtr.append(v.transform(train["text"])); Xte.append(v.transform(test["text"]))
    stacks[tag] = (hstack([centre(x, train["prompt_id"].values) for x in Xtr]).tocsr(), Xte)

wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), sublinear_tf=True, min_df=2, max_features=100000)
H = wv.fit_transform(test["text"])
ks = (24, 30, 36)
lab = {k: KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(H) for k in ks}

def acc(P):
    return balanced_accuracy_score(truth, [MODELS[i] for i in np.argmax(P, 1)])

for tag, (A, Xte) in stacks.items():
    for C in (10, 30, 100):
        clf = LogisticRegression(C=C, max_iter=3000).fit(A, train["model"].values)
        order = [list(clf.classes_).index(m) for m in MODELS]
        probs = []
        for k in ks:
            B = hstack([centre(x, lab[k]) for x in Xte]).tocsr()
            probs.append(clf.predict_proba(B)[:, order])
        print(f"{tag:5s} C={C:<4} bal_acc={acc(np.mean(probs, 0)):.4f}", flush=True)
