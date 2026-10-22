# Screening: is there signal in the forum corpora beyond what we already train on?
# Cheap linear probe, prompt-grouped holdout from the official train only.
# A = train only, B = train+pool_v1 (what the LoRA recipe already uses),
# C = B + marginal, D = train + marginal only.
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

CFGS = [
    dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=3, max_features=200000),
    dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3, max_features=300000),
]


def main():
    t0 = time.time()
    tr = pd.read_csv("data/train.csv")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    a, b = next(gss.split(tr, tr["model"], groups=tr["prompt_id"]))
    train, hold = tr.iloc[a].reset_index(drop=True), tr.iloc[b].reset_index(drop=True)

    uni = pd.read_parquet("extra/union_all.parquet")
    pool = uni[uni.in_pool_v1]
    marg = uni[(~uni.in_train) & (~uni.in_pool_v1)]
    hold_h = set(hold["text"].str.lower().str.strip())
    marg = marg[~marg["text"].str.lower().str.strip().isin(hold_h)]
    pool = pool[~pool["text"].str.lower().str.strip().isin(hold_h)]
    print(f"train {len(train)} | holdout {len(hold)} | pool {len(pool)} | marginal {len(marg)}")

    corpus = pd.concat([train[["text"]], pool[["text"]], marg[["text"]]], ignore_index=True)
    vs = [TfidfVectorizer(**cfg).fit(corpus["text"]) for cfg in CFGS]
    print(f"vectorisers fitted on {len(corpus)} docs in {time.time()-t0:.0f}s")

    def feats(df):
        return hstack([v.transform(df["text"]) for v in vs]).tocsr()

    Xh = feats(hold)
    cache = {}

    def get(df, key):
        if key not in cache:
            cache[key] = feats(df)
        return cache[key]

    arms = {
        "A train only": train,
        "B train+pool": pd.concat([train, pool], ignore_index=True),
        "C train+pool+marg": pd.concat([train, pool, marg], ignore_index=True),
        "D train+marg": pd.concat([train, marg], ignore_index=True),
    }
    rows = []
    for name, fit in arms.items():
        X = get(fit, name)
        for C in (10, 100):
            clf = LogisticRegression(C=C, max_iter=1500, class_weight="balanced").fit(X, fit["model"])
            p = clf.predict(Xh)
            rows.append(dict(arm=name, C=C, n=len(fit),
                             bal_acc=balanced_accuracy_score(hold["model"], p),
                             macro_f1=f1_score(hold["model"], p, average="macro")))
            print(f"{name:20s} C={C:4d} n={len(fit):6d} bal_acc={rows[-1]['bal_acc']:.4f} "
                  f"macroF1={rows[-1]['macro_f1']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv("extra/screen_tfidf.csv", index=False)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
