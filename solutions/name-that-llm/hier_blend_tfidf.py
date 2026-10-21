"""Can the k=3 style-cluster prior be built from TF-IDF alone (no embeddings, pure sklearn)?

If yes, the CPU-only notebook can carry the geo correction without shipping 4B weights.
Compares, per seed: tfidf flat vs tfidf flat + alpha*log P_group (level-1 trained on the same TF-IDF block).
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
for m in MODELS[:5] + ["qwen3.8-max", "kimi-k3"]:
    G3[m] = "gptish"
for m in ["gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"]:
    G3[m] = "gemini"
for m in ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
          "glm-5.3", "glm-5.3-flash"]:
    G3[m] = "claudeish"
GS = ["gemini", "gptish", "claudeish"]
GOF = np.array([GS.index(G3[m]) for m in MODELS])
CFGS = [dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000),
        dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000),
        dict(analyzer="char_wb", ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_features=500000)]


def acc(y, P):
    return balanced_accuracy_score(y, [MODELS[i] for i in np.argmax(P, 1)])


def main():
    tr = pd.read_csv("data/train.csv")
    ym = tr["model"].values
    rows = []
    for seed in (42, 7, 2024):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    .split(tr, ym, groups=tr["prompt_id"]))
        train, hold = tr.iloc[a].reset_index(drop=True), tr.iloc[b].reset_index(drop=True)
        ytr, yho = ym[a], ym[b]
        Xtr, Xho = [], []
        for cfg in CFGS:
            v = TfidfVectorizer(**cfg).fit(train["text"])
            Xtr.append(v.transform(train["text"]))
            Xho.append(v.transform(hold["text"]))
        sc = StandardScaler().fit(style_features(train["text"]))
        A = hstack(Xtr + [csr_matrix(sc.transform(style_features(train["text"])) * 0.3)]).tocsr()
        B = hstack(Xho + [csr_matrix(sc.transform(style_features(hold["text"])) * 0.3)]).tocsr()

        flat = LogisticRegression(C=100, max_iter=3000).fit(A, ytr)
        P = flat.predict_proba(B)[:, [list(flat.classes_).index(m) for m in MODELS]]

        gtr = np.array([G3[m] for m in ytr])
        l1 = LogisticRegression(C=100, max_iter=3000, class_weight="balanced").fit(A, gtr)
        P_g = l1.predict_proba(B)[:, [list(l1.classes_).index(g) for g in GS]]
        l1acc = balanced_accuracy_score([G3[m] for m in yho], [GS[i] for i in P_g.argmax(1)])

        Lp = np.log(np.clip(P, 1e-9, 1))
        out = {}
        for alpha in (0.2, 0.3, 0.5, 0.7):
            L = Lp + alpha * np.log(np.clip(P_g, 1e-9, 1))[:, GOF]
            out[alpha] = acc(yho, np.exp(L - L.max(1, keepdims=True)))
        print(f"seed {seed:5d} flat={acc(yho, P):.4f} l1={l1acc:.3f} | "
              + " ".join(f"a{k}:{v:.4f}" for k, v in out.items()), flush=True)
        rows.append([acc(yho, P)] + list(out.values()))
    r = np.array(rows)
    names = ["flat", "geo0.2", "geo0.3", "geo0.5", "geo0.7"]
    print("\n" + "  ".join(f"{n}={r[:, i].mean():.4f}" for i, n in enumerate(names)))
    for i, n in enumerate(names[1:], 1):
        print(f"{n} - flat: {np.round(r[:, i] - r[:, 0], 4)}")


if __name__ == "__main__":
    main()
