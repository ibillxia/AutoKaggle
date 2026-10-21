"""Does the k=3 style-cluster hierarchy add anything on top of the best 3-way blend (0.4315)?

base  = log-mean of TF-IDF/style + 4B embeddings + 9B embeddings  (weights 0.35/0.35/0.30)
base+geo(a)   = log P_base + a * log P_group   (level-1 posterior of the k=3 style clusters)
base+hier(b)  = (1-b) log P_base + b log P_softcascade

Groups G3 (from the OOF-confusion clustering):
  gemini family | gpt + qwen + kimi | claude + glm
Every variant is scored as 18-class balanced accuracy on 5 different grouped holdouts.
"""
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
from test_style_features import style_features

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
G3 = {}
for m in ["gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"]:
    G3[m] = "gemini"
for m in ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
          "qwen3.8-max", "kimi-k3"]:
    G3[m] = "gptish"
for m in ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
          "glm-5.3", "glm-5.3-flash"]:
    G3[m] = "claudeish"
GS = ["gemini", "gptish", "claudeish"]
GOF = np.array([GS.index(G3[m]) for m in MODELS])
CFGS = [dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000),
        dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000),
        dict(analyzer="char_wb", ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_features=500000)]
W = (0.35, 0.35, 0.30)


def acc(y, P):
    return balanced_accuracy_score(y, [MODELS[i] for i in np.argmax(P, 1)])


def order_of(clf, classes):
    return [list(clf.classes_).index(c) for c in classes]


def tfidf_probs(train, hold, ytr):
    Xtr, Xho = [], []
    for cfg in CFGS:
        v = TfidfVectorizer(**cfg).fit(train["text"])
        Xtr.append(v.transform(train["text"]))
        Xho.append(v.transform(hold["text"]))
    sc = StandardScaler().fit(style_features(train["text"]))
    A = hstack(Xtr + [csr_matrix(sc.transform(style_features(train["text"])) * 0.3)]).tocsr()
    B = hstack(Xho + [csr_matrix(sc.transform(style_features(hold["text"])) * 0.3)]).tocsr()
    clf = LogisticRegression(C=100, max_iter=3000).fit(A, ytr)
    return clf.predict_proba(B)[:, order_of(clf, MODELS)]


def emb_probs(X, a, b, ytr, C=1.0):
    sc = StandardScaler().fit(X[a])
    clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[a]).astype(np.float32), ytr)
    return clf.predict_proba(sc.transform(X[b]).astype(np.float32))[:, order_of(clf, MODELS)]


def load_emb(path):
    d = np.load(path, allow_pickle=True)
    layers = sorted(int(k[1:]) for k in d.files if k.startswith("L"))
    return np.concatenate([d[f"L{L}"] for L in layers], axis=1).astype(np.float32), d["y"]


def level1(X, a, b, ytr):
    """k=3 style-cluster posterior from the 4B space, plus the soft cascade built on it."""
    sc = StandardScaler().fit(X[a])
    Xtr, Xho = sc.transform(X[a]).astype(np.float32), sc.transform(X[b]).astype(np.float32)
    gtr = np.array([G3[m] for m in ytr])
    l1 = LogisticRegression(C=100, max_iter=3000, class_weight="balanced").fit(Xtr, gtr)
    P_g = l1.predict_proba(Xho)[:, order_of(l1, GS)]
    P_cond = np.zeros((len(Xho), len(MODELS)))
    for g in GS:
        cols = [i for i, m in enumerate(MODELS) if G3[m] == g]
        mask = gtr == g
        sub = LogisticRegression(C=100, max_iter=3000, class_weight="balanced").fit(Xtr[mask], ytr[mask])
        P_cond[:, cols] = sub.predict_proba(Xho)[:, order_of(sub, [MODELS[i] for i in cols])]
    soft = P_g[:, GOF] * P_cond
    soft /= np.clip(soft.sum(1, keepdims=True), 1e-12, None)
    return P_g, soft


def main():
    tr = pd.read_csv("data/train.csv")
    ym = tr["model"].values
    X4, y4 = load_emb("emb_qwen4b_train.npz")
    X9, y9 = load_emb("emb_qwen9b_train.npz")
    assert (y4 == ym).all() and (y9 == ym).all()

    rows = []
    for seed in (42, 7, 2024, 13, 99):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    .split(tr, ym, groups=tr["prompt_id"]))
        train, hold = tr.iloc[a].reset_index(drop=True), tr.iloc[b].reset_index(drop=True)
        ytr, yho = ym[a], ym[b]

        P_tf = tfidf_probs(train, hold, ytr)
        P4 = emb_probs(X4, a, b, ytr)
        P9 = emb_probs(X9, a, b, ytr)
        Lb = sum(w * np.log(np.clip(P, 1e-9, 1)) for w, P in zip(W, (P_tf, P4, P9)))
        P_base = np.exp(Lb - Lb.max(1, keepdims=True))
        P_base /= P_base.sum(1, keepdims=True)

        P_g, soft = level1(X4, a, b, ytr)
        Lg = np.log(np.clip(P_g, 1e-9, 1))[:, GOF]
        geo = {}
        for alpha in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            L = Lb + alpha * Lg
            geo[alpha] = acc(yho, np.exp(L - L.max(1, keepdims=True)))
        hier = {}
        Ls = np.log(np.clip(soft, 1e-9, 1))
        for beta in (0.2, 0.3, 0.4, 0.5):
            L = (1 - beta) * Lb + beta * Ls
            hier[beta] = acc(yho, np.exp(L - L.max(1, keepdims=True)))

        base = acc(yho, P_base)
        print(f"seed {seed:5d} base={base:.4f} (tf={acc(yho, P_tf):.3f} 4b={acc(yho, P4):.3f} 9b={acc(yho, P9):.3f})"
              f" | geo " + " ".join(f"{k}:{v:.4f}" for k, v in geo.items())
              + " | hier " + " ".join(f"{k}:{v:.4f}" for k, v in hier.items()), flush=True)
        rows.append([base] + [geo[k] for k in (0.2, 0.3, 0.5)] + [hier[0.3], hier[0.5]])

    r = np.array(rows)
    names = ["base", "geo0.2", "geo0.3", "geo0.5", "hier0.3", "hier0.5"]
    print("\n" + "  ".join(f"{n}={r[:, i].mean():.4f}±{r[:, i].std():.4f}" for i, n in enumerate(names)))
    for i, n in enumerate(names[1:], start=1):
        d = r[:, i] - r[:, 0]
        print(f"{n} - base: mean={d.mean():+.4f}  per-seed={np.round(d, 4)}")


if __name__ == "__main__":
    main()
