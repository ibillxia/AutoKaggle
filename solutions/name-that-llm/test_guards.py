"""Measure the two post-processing guards of the 0.604 notebook on a REAL out-of-fold matrix.

`dariushafshar/ntl-oof-logits-one-public-lora24-run` ships prompt-disjoint OOF logits for all
2,699 train rows (plain LoRA24 head, balanced accuracy 0.4804). That is enough to answer, with
labels and no leaderboard, the questions the hybrid kernel currently answers by assumption:

  1. does Sinkhorn toward a uniform class prior help when the real prior is unknown?
  2. does the spaced em-dash guard help, and how many rows does it touch?
  3. does the dense-cluster Hungarian guard help -- with *true* prompt groups (oracle upper
     bound) and with the tfidf@0.85 clusters the notebook actually recovers?

The head is weaker than ours (0.48 vs 0.60), so treat magnitudes as directional. Direction is
what decides whether the guard belongs in the blend at all.
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score

TRAIN = "../../data/ntl/train.csv"
OOF = "oof/oof_logits_runB_lucasboesen_lora24_s7.npy"

train = pd.read_csv(TRAIN)
want = open("oof/train_row_alignment.txt").read().split()[-1]
got = hashlib.sha256("\n".join(train["id"]).encode()).hexdigest()
assert got == want, "row order does not match the OOF matrix"
L = np.load(OOF)
classes = json.load(open("oof/classes.json"))
y = train["model"].map({c: i for i, c in enumerate(classes)}).to_numpy()
assert L.shape == (len(train), 18), L.shape
K = 18
texts = train["text"].astype(str).to_numpy()
prompt = train["prompt_id"].to_numpy()


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def ba(P):
    return balanced_accuracy_score(y, P.argmax(1))


def sinkhorn(P, iters=100):
    P = P.copy()
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        P *= target / P.sum(0)
        P /= P.sum(1, keepdims=True)
    return P


SPACED_ALLOWED = {"claude-opus-4-6", "claude-opus-5", "glm-5.3", "glm-5.3-flash"}
forbidden = np.array([c not in SPACED_ALLOWED for c in classes])


def emdash_guard(P, texts, verbose=False):
    P = P.copy()
    hit = np.array([" \u2014 " in t for t in texts])
    fixed = 0
    for i in np.where(hit)[0]:
        if classes[P[i].argmax()] not in SPACED_ALLOWED:
            P[i, forbidden] = 0.0
            P[i] /= P[i].sum()
            fixed += 1
    if verbose:
        print(f"    em-dash guard: {int(hit.sum())} rows carry ' \u2014 ', corrected {fixed}")
    return P


def hungarian_guard(P, groups, lo=12, hi=18, verbose=False):
    P = P.copy()
    changes = 0
    applied = 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if not (lo <= len(idx) <= hi):
            continue
        applied += len(idx)
        cost = -np.log(np.clip(P[idx, :], 1e-12, 1.0))
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if P[idx[r]].argmax() != c:
                changes += 1
        for r, c in zip(rows, cols):
            P[idx[r], :] = 0.0
            P[idx[r], c] = 1.0
    if verbose:
        print(f"    hungarian: {applied} rows in clusters {lo}-{hi}, {changes} one-hot flips")
    return P


def pairwise_f1(groups, truth):
    n = len(groups)
    agree = np.zeros((n, n), bool)
    same_pred = groups[:, None] == groups[None, :]
    same_true = truth[:, None] == truth[None, :]
    np.fill_diagonal(same_pred, False)
    np.fill_diagonal(same_true, False)
    tp = (same_pred & same_true).sum()
    fp = (same_pred & ~same_true).sum()
    fn = (~same_pred & same_true).sum()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9), len(np.unique(groups))


def tfidf_groups(texts, thresh=0.85, method="average", k=None):
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(list(texts))
    D = np.clip(1 - (X @ X.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    Dc = squareform(D, checks=False)
    if k is not None:
        return fcluster(linkage(Dc, method=method), k, criterion="maxclust")
    return fcluster(linkage(Dc, method=method), thresh, criterion="distance")


print("=" * 74)
P0 = softmax(L)
print(f"baseline                      ba={ba(P0):.4f}")

# ---- Sinkhorn ---------------------------------------------------------------
P_sk = sinkhorn(P0)
print(f"+ sinkhorn(uniform)           ba={ba(P_sk):.4f}  ({ba(P_sk)-ba(P0):+.4f})")

# ---- em-dash ---------------------------------------------------------------
print("\n-- em-dash guard --")
P_em = emdash_guard(P0, texts, verbose=True)
print(f"+ em-dash                    ba={ba(P_em):.4f}  ({ba(P_em)-ba(P0):+.4f})")
P_sk_em = emdash_guard(P_sk, texts, verbose=True)
print(f"+ sinkhorn + em-dash         ba={ba(P_sk_em):.4f}  ({ba(P_sk_em)-ba(P0):+.4f})")

# ---- grouping quality ------------------------------------------------------
print("\n-- recovered groups vs true prompts --")
true_groups = pd.factorize(prompt)[0] + 1
for thresh in (0.80, 0.85, 0.90):
    g = tfidf_groups(texts, thresh=thresh)
    f1, k = pairwise_f1(g, true_groups)
    big = sum(1 for c in np.bincount(g) if 12 <= c <= 18)
    print(f"    tfidf avg@{thresh:.2f}: {k} clusters, pairwise F1={f1:.4f}, clusters of 12-18: {big}")
g_ward = tfidf_groups(texts, method="ward", k=150)
f1w, _ = pairwise_f1(g_ward, true_groups)
print(f"    ward k=150: pairwise F1={f1w:.4f}")

# ---- Hungarian ------------------------------------------------------------
print("\n-- hungarian guard (clusters 12-18) --")
for name, g in (("oracle prompt groups", true_groups),
                ("tfidf avg@0.85", tfidf_groups(texts, thresh=0.85))):
    for label, base in (("plain", P0), ("after sinkhorn+emdash", P_sk_em)):
        P_h = hungarian_guard(base, g)
        print(f"    {name:22s} on {label:22s} ba={ba(P_h):.4f}  ({ba(P_h)-ba(base):+.4f})")

# ---- what the notebook ships ----------------------------------------------
print("\n-- shipped pipeline: sinkhorn -> em-dash -> hungarian(tfidf@0.85) --")
P_ship = hungarian_guard(P_sk_em, tfidf_groups(texts, thresh=0.85))
print(f"    shipped ba={ba(P_ship):.4f}  vs baseline {ba(P0):.4f}")

# ---- ceiling: perfect groups, strong constraint ---------------------------
print("\n-- reference: oracle groups, all cluster sizes --")
for lo, hi in ((2, 18), (12, 18), (17, 18)):
    P_h = hungarian_guard(P0, true_groups, lo=lo, hi=hi)
    print(f"    hungarian oracle {lo}-{hi}: ba={ba(P_h):.4f}  ({ba(P_h)-ba(P0):+.4f})")
