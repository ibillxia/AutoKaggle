"""Two-level (vendor -> version) classification vs flat 18-way, on the same grouped holdout.

Level 1: 6 vendors (openai / google / anthropic / alibaba / zhipu / moonshot)
Level 2: version inside the vendor

Compared variants (all scored as 18-class balanced accuracy):
  flat            - single 18-way logistic regression (current baseline)
  vendor6         - level-1 only, reported as 6-class balanced accuracy + per-vendor recall
  cascade_hard    - argmax vendor, then argmax version within it
  flat_masked     - flat probabilities with non-predicted vendors zeroed, then argmax
  cascade_soft    - P(vendor) * P(version | vendor), renormalised
  cascade_oracle  - level-2 with the TRUE vendor (upper bound of the hierarchy)
  geo(a)          - log P_flat + a * log P_vendor[true-vendor-of-class]
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from test_style_features import style_features

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
VENDOR = {
    "openai": ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"],
    "google": ["gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
    "anthropic": ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5"],
    "alibaba": ["qwen3.8-max"],
    "zhipu": ["glm-5.3", "glm-5.3-flash"],
    "moonshot": ["kimi-k3"],
}
VENDORS = sorted(VENDOR)
VMAP = {m: v for v, ms in VENDOR.items() for m in ms}
VOF = np.array([VENDORS.index(VMAP[m]) for m in MODELS])          # vendor index per model column
CFGS = [dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000),
        dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000),
        dict(analyzer="char_wb", ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_features=500000)]


def aligned(clf, X, classes):
    """predict_proba reordered to the given class order."""
    order = [list(clf.classes_).index(c) for c in classes]
    return clf.predict_proba(X)[:, order]


def acc18(truth, P):
    return balanced_accuracy_score(truth, [MODELS[i] for i in np.argmax(P, 1)])


def build_feats(train, hold, feat):
    if feat == "tfidf":
        Xtr, Xho = [], []
        for cfg in CFGS:
            v = TfidfVectorizer(**cfg).fit(train["text"])
            Xtr.append(v.transform(train["text"]))
            Xho.append(v.transform(hold["text"]))
        sc = StandardScaler().fit(style_features(train["text"]))
        A = hstack(Xtr + [csr_matrix(sc.transform(style_features(train["text"])) * 0.3)]).tocsr()
        B = hstack(Xho + [csr_matrix(sc.transform(style_features(hold["text"])) * 0.3)]).tocsr()
        return A, B
    if feat == "emb":
        d = np.load("emb_qwen4b_train.npz", allow_pickle=True)
        layers = sorted(int(k[1:]) for k in d.files if k.startswith("L"))
        X = np.concatenate([d[f"L{L}"] for L in layers], axis=1).astype(np.float32)
        sc = StandardScaler().fit(X)
        X = sc.transform(X).astype(np.float32)
        tr_idx = np.load("emb_split_train_idx.npy") if False else None  # split is deterministic below
        return X, None
    raise ValueError(feat)


def run(feat, C_flat, C_vendor, C_inner):
    tr = pd.read_csv("data/train.csv")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    a, b = next(gss.split(tr, tr["model"], groups=tr["prompt_id"]))
    train, hold = tr.iloc[a].reset_index(drop=True), tr.iloc[b].reset_index(drop=True)
    ytr, yho = train["model"].values, hold["model"].values

    if feat == "emb":
        d = np.load("emb_qwen4b_train.npz", allow_pickle=True)
        layers = sorted(int(k[1:]) for k in d.files if k.startswith("L"))
        Xall = np.concatenate([d[f"L{L}"] for L in layers], axis=1).astype(np.float32)
        assert (d["y"] == tr["model"].values).all(), "embedding row order mismatch"
        sc = StandardScaler().fit(Xall[a])
        Xtr, Xho = sc.transform(Xall[a]).astype(np.float32), sc.transform(Xall[b]).astype(np.float32)
    else:
        Xtr, Xho = build_feats(train, hold, feat)

    # ---------------- level 0: flat 18-way ----------------
    flat = LogisticRegression(C=C_flat, max_iter=3000).fit(Xtr, ytr)
    P_flat = aligned(flat, Xho, MODELS)

    # ---------------- level 1: 6 vendors ----------------
    vtr = np.array([VMAP[m] for m in ytr])
    vend = LogisticRegression(C=C_vendor, max_iter=3000, class_weight="balanced").fit(Xtr, vtr)
    P_v = aligned(vend, Xho, VENDORS)
    print(f"[{feat}] vendor6 acc = {balanced_accuracy_score([VMAP[m] for m in yho], [VENDORS[i] for i in P_v.argmax(1)]):.4f}")
    print("     per-vendor recall: " + ", ".join(
        f"{v}={confusion_matrix([VMAP[m] for m in yho], [VENDORS[i] for i in P_v.argmax(1)], labels=VENDORS)[k, k] / max(1, (np.array([VMAP[m] for m in yho]) == v).sum()):.2f}"
        for k, v in enumerate(VENDORS)))

    # ---------------- level 2: version inside each vendor ----------------
    P_cond = np.zeros((len(hold), len(MODELS)), dtype=np.float64)
    for v in VENDORS:
        cols = [MODELS.index(m) for m in VENDOR[v]]
        mask = (vtr == v)
        if len(VENDOR[v]) == 1 or mask.sum() < 20:
            P_cond[:, cols] = 1.0
            continue
        sub = LogisticRegression(C=C_inner, max_iter=3000, class_weight="balanced")
        sub.fit(Xtr[mask], ytr[mask])
        P_cond[:, cols] = aligned(sub, Xho, VENDOR[v])

    # ---------------- variants ----------------
    res = {}
    res["flat"] = acc18(yho, P_flat)

    vhat = P_v.argmax(1)
    # hard cascade
    Pc = P_cond.copy()
    keep = np.zeros_like(Pc, dtype=bool)
    for i in range(len(hold)):
        keep[i, VOF == vhat[i]] = True
    Pc = np.where(keep, Pc, 0.0)
    res["cascade_hard"] = acc18(yho, Pc)
    # flat masked by predicted vendor
    Pm = np.where(keep, P_flat, 0.0)
    res["flat_masked"] = acc18(yho, Pm)
    # soft hierarchy
    Psoft = P_v[:, VOF] * P_cond
    Psoft /= np.clip(Psoft.sum(1, keepdims=True), 1e-12, None)
    res["cascade_soft"] = acc18(yho, Psoft)
    # oracle vendor (upper bound of the two-level structure)
    vtrue = np.array([VENDORS.index(VMAP[m]) for m in yho])
    ko = np.zeros_like(P_cond, dtype=bool)
    for i in range(len(hold)):
        ko[i, VOF == vtrue[i]] = True
    res["cascade_oracle"] = acc18(yho, np.where(ko, P_cond, 0.0))

    # geometric reweighting of flat probs by vendor posterior
    Lp, Lv = np.log(np.clip(P_flat, 1e-9, 1)), np.log(np.clip(P_v, 1e-9, 1))[:, VOF]
    geo = {}
    for alpha in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        geo[alpha] = acc18(yho, Lp + alpha * Lv)

    print(f"--- {feat} (C_flat={C_flat}, C_vendor={C_vendor}, C_inner={C_inner})")
    for k, v in res.items():
        print(f"    {k:16s} {v:.4f}")
    print("    geo alpha sweep  " + " ".join(f"{a}:{s:.4f}" for a, s in geo.items()))
    np.savez(f"hier_probs_{feat}.npz", P_flat=P_flat, P_v=P_v, P_cond=P_cond, y=yho)
    return res, geo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="tfidf", choices=["tfidf", "emb"])
    ap.add_argument("--C_flat", type=float, default=100.0)
    ap.add_argument("--C_vendor", type=float, default=100.0)
    ap.add_argument("--C_inner", type=float, default=100.0)
    a = ap.parse_args()
    run(a.feat, a.C_flat, a.C_vendor, a.C_inner)
