"""Dump per-seed probabilities for the shippable TF-IDF + 4B configuration, then grid-search
blend weight / geo strength / 4B regularisation on the 5-seed mean (not a single holdout).
"""
import itertools

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
from style_feat import style_features

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
G3 = {}
for m in MODELS[:5] + ["qwen3.8-max", "kimi-k3"]:
    G3[m] = "gptish"
for m in MODELS[5:9]:
    G3[m] = "gemini"
for m in MODELS[9:14] + ["glm-5.3", "glm-5.3-flash"]:
    G3[m] = "claudeish"
GS = ["gemini", "gptish", "claudeish"]
GOF = np.array([GS.index(G3[m]) for m in MODELS])
CFGS = [dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000),
        dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000),
        dict(analyzer="char_wb", ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_features=500000)]
SEEDS = (42, 7, 2024, 13, 99)


def main():
    tr = pd.read_csv("data/train.csv")
    ym = tr["model"].values
    d = np.load("emb_qwen4b_train.npz", allow_pickle=True)
    layers = sorted(int(k[1:]) for k in d.files if k.startswith("L"))
    E = np.concatenate([d[f"L{L}"] for L in layers], axis=1).astype(np.float32)
    assert (d["y"] == ym).all()

    store = {}
    for seed in SEEDS:
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
        store[f"tf_{seed}"] = tf.predict_proba(B)[:, [list(tf.classes_).index(m) for m in MODELS]]

        sc_e = StandardScaler().fit(E[a])
        Etr, Eho = sc_e.transform(E[a]).astype(np.float32), sc_e.transform(E[b]).astype(np.float32)
        for C in (1.0, 10.0):
            em = LogisticRegression(C=C, max_iter=2000).fit(Etr, ytr)
            store[f"emb{C:g}_{seed}"] = em.predict_proba(Eho)[:, [list(em.classes_).index(m) for m in MODELS]]

        gtr = np.array([G3[m] for m in ytr])
        l1 = LogisticRegression(C=100, max_iter=3000, class_weight="balanced").fit(A, gtr)
        store[f"g_{seed}"] = l1.predict_proba(B)[:, [list(l1.classes_).index(g) for g in GS]]
        store[f"y_{seed}"] = yho
        print(f"seed {seed} probs dumped", flush=True)

    np.savez("final_probs_5seed.npz", **store)

    # ---- grid search on the 5-seed mean ----
    def score(w4, alpha, C):
        vals = []
        for seed in SEEDS:
            P_tf, P_e, P_g = store[f"tf_{seed}"], store[f"emb{C:g}_{seed}"], store[f"g_{seed}"]
            L = (1 - w4) * np.log(np.clip(P_tf, 1e-9, 1)) + w4 * np.log(np.clip(P_e, 1e-9, 1))
            if alpha:
                L = L + alpha * np.log(np.clip(P_g, 1e-9, 1))[:, GOF]
            vals.append(balanced_accuracy_score(store[f"y_{seed}"], [MODELS[i] for i in np.argmax(L, 1)]))
        return float(np.mean(vals)), float(np.std(vals))

    best = None
    print("\n w4   alpha  C     mean±std")
    for w4, alpha, C in itertools.product((0.4, 0.5, 0.6, 0.7, 0.8), (0.0, 0.2, 0.3, 0.4, 0.5, 0.7), (1.0, 10.0)):
        m, s = score(w4, alpha, C)
        tag = "*" if best is None or m > best[0] else " "
        if best is None or m > best[0]:
            best = (m, s, w4, alpha, C)
        print(f"{tag} {w4:.1f}  {alpha:.1f}   {C:<4g}  {m:.4f}±{s:.4f}")
    print(f"\nbest: w4={best[2]} alpha={best[3]} C={best[4]} -> {best[0]:.4f}±{best[1]:.4f}")


if __name__ == "__main__":
    main()
