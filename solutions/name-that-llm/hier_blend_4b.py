"""CV of the shippable CPU+4B variant: TF-IDF/style + 4B embeddings + k=3 group prior.

9B embeddings do not fit in a single Kaggle dataset together with 4B, so this is the
"one dataset, fp16 on a single T4" configuration. Checks how much the +geo term still gives.
"""
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, ".")
from style_feat import style_features
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
G3 = {}
for m in ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
          "qwen3.8-max", "kimi-k3"]:
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
    d = np.load("emb_qwen4b_train.npz", allow_pickle=True)
    layers = sorted(int(k[1:]) for k in d.files if k.startswith("L"))
    E = np.concatenate([d[f"L{L}"] for L in layers], axis=1).astype(np.float32)
    assert (d["y"] == ym).all()

    for seed in (42, 7, 2024, 13, 99):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    .split(tr, ym, groups=tr["prompt_id"]))
        train, hold = tr.iloc[a].reset_index(drop=True), tr.iloc[b].reset_index(drop=True)
        ytr, yho = ym[a], ym[b]

        Xtr, Xho = [], []
        for cfg in CFGS:
            v = TfidfVectorizer(**cfg).fit(train["text"])
            Xtr.append(v.transform(train["text"]))
            Xho.append(v.transform(hold["text"]))
        sc_s = StandardScaler().fit(style_features(train["text"]))
        A = hstack(Xtr + [csr_matrix(sc_s.transform(style_features(train["text"])) * 0.3)]).tocsr()
        B = hstack(Xho + [csr_matrix(sc_s.transform(style_features(hold["text"])) * 0.3)]).tocsr()

        tf = LogisticRegression(C=100, max_iter=3000).fit(A, ytr)
        P_tf = tf.predict_proba(B)[:, [list(tf.classes_).index(m) for m in MODELS]]

        sc_e = StandardScaler().fit(E[a])
        Etr, Eho = sc_e.transform(E[a]).astype(np.float32), sc_e.transform(E[b]).astype(np.float32)
        em = LogisticRegression(C=1.0, max_iter=2000).fit(Etr, ytr)
        P_4b = em.predict_proba(Eho)[:, [list(em.classes_).index(m) for m in MODELS]]

        gtr = np.array([G3[m] for m in ytr])
        l1 = LogisticRegression(C=100, max_iter=3000, class_weight="balanced").fit(A, gtr)
        P_g = l1.predict_proba(B)[:, [list(l1.classes_).index(g) for g in GS]]
        l1acc = balanced_accuracy_score([G3[m] for m in yho], [GS[i] for i in P_g.argmax(1)])

        Lb = 0.5 * np.log(np.clip(P_tf, 1e-9, 1)) + 0.5 * np.log(np.clip(P_4b, 1e-9, 1))
        base = acc(yho, np.exp(Lb - Lb.max(1, keepdims=True)))
        Lg = np.log(np.clip(P_g, 1e-9, 1))[:, GOF]
        out = {}
        for al in (0.2, 0.3, 0.5, 0.7):
            L = Lb + al * Lg
            out[al] = acc(yho, np.exp(L - L.max(1, keepdims=True)))
        print(f"seed {seed:5d} tf={acc(yho, P_tf):.4f} 4b={acc(yho, P_4b):.4f} l1={l1acc:.3f}"
              f" base={base:.4f} | " + " ".join(f"a{k}:{v:.4f}" for k, v in out.items()), flush=True)


if __name__ == "__main__":
    main()
