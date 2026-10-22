"""How much of the Hungarian guard's gain survives a realistically recovered grouping?

test_guards.py showed, on real OOF logits (balanced accuracy 0.4804):
  oracle prompt groups  -> 0.6783  (+0.1978)
  tfidf average @0.85   -> 0.6301  (+0.1497)

So the guard is real and its size is set almost entirely by grouping quality. This script
sweeps grouping recipes and guard softening on the same OOF, still CPU-only, to pick the
variant for the next hybrid kernel.

Two things matter and they pull in opposite directions:
  - tighter clusters (higher threshold / ward) recover prompts better -> bigger gain,
  - but a cluster that accidentally merges two prompts must not be solved as one assignment,
    which is what the 12..18 size window and a soft update are for.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score

train = pd.read_csv("../../data/ntl/train.csv")
want = open("oof/train_row_alignment.txt").read().split()[-1]
assert hashlib.sha256("\n".join(train["id"]).encode()).hexdigest() == want
L = np.load("oof/oof_logits_runB_lucasboesen_lora24_s7.npy")
classes = json.load(open("oof/classes.json"))
y = train["model"].map({c: i for i, c in enumerate(classes)}).to_numpy()
texts = train["text"].astype(str).tolist()
truth = pd.factorize(train["prompt_id"])[0] + 1
K, n = 18, len(train)


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def ba(P):
    return balanced_accuracy_score(y, P.argmax(1))


def pairwise_f1(groups):
    same_p = groups[:, None] == groups[None, :]
    same_t = truth[:, None] == truth[None, :]
    np.fill_diagonal(same_p, False)
    np.fill_diagonal(same_t, False)
    tp = (same_p & same_t).sum()
    fp = (same_p & ~same_t).sum()
    fn = (~same_p & same_t).sum()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9)


X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
D = np.clip(1 - (X @ X.T).toarray(), 0, None)
np.fill_diagonal(D, 0)
Dc = squareform(D, checks=False)

GROUPS = {}
for th in (0.82, 0.85, 0.88, 0.90, 0.92, 0.95):
    GROUPS[f"avg@{th:.2f}"] = fcluster(linkage(Dc, method="average"), th, criterion="distance")
for th in (0.90, 0.95, 1.00):
    GROUPS[f"complete@{th:.2f}"] = fcluster(linkage(Dc, method="complete"), th, criterion="distance")
for k in (150, 160):
    GROUPS[f"ward k={k}"] = fcluster(linkage(Dc, method="ward"), k, criterion="maxclust")
GROUPS["sklearn-ward n/18"] = AgglomerativeClustering(
    n_clusters=K and int(round(n / 18)), linkage="ward", metric="euclidean"
).fit_predict(X.toarray())
GROUPS["oracle"] = truth

P0 = softmax(L)


def apply_hungarian(P, groups, lo=12, hi=18, soft=False, damp=1.0):
    """soft=True keeps the original distribution but forbids columns that the assignment
    gave to another row of the same cluster -- robust when a cluster merges two prompts."""
    P = P.copy()
    flips = 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if not (lo <= len(idx) <= hi):
            continue
        cost = -np.log(np.clip(P[idx, :], 1e-12, 1.0))
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if P[idx[r]].argmax() != c:
                flips += 1
        if soft:
            keep = np.zeros((len(idx), K), bool)
            keep[rows, cols] = True
            Q = np.where(keep, P[idx] ** damp, 0.0)
            s = Q.sum(1, keepdims=True)
            bad = s[:, 0] <= 0
            Q[bad] = keep[bad] / keep[bad].sum(1, keepdims=True)
            P[idx] = Q / Q.sum(1, keepdims=True)
        else:
            for r, c in zip(rows, cols):
                P[idx[r], :] = 0.0
                P[idx[r], c] = 1.0
    return P, flips


print(f"{'grouping':22s} {'F1':>6s} {'rows12-18':>9s} {'hard':>7s} {'soft':>7s} {'flips':>6s}")
rows = []
for name, g in GROUPS.items():
    f1 = pairwise_f1(g)
    sizes = np.bincount(g)
    cov = int(sum(c for c in sizes if 12 <= c <= 18))
    Ph, fl = apply_hungarian(P0, g)
    Ps, _ = apply_hungarian(P0, g, soft=True)
    rows.append((name, f1, cov, ba(Ph), ba(Ps), fl))
    print(f"{name:22s} {f1:6.4f} {cov:9d} {ba(Ph):7.4f} {ba(Ps):7.4f} {fl:6d}   "
          f"(hard {ba(Ph)-ba(P0):+.4f}, soft {ba(Ps)-ba(P0):+.4f})")

print(f"\nbaseline {ba(P0):.4f}")

# size window sweep on the best grouping
best = max(rows, key=lambda r: r[3])
print(f"\nsize-window sweep on {best[0]} (hard):")
g = GROUPS[best[0]]
for lo, hi in ((2, 18), (8, 18), (10, 18), (12, 18), (14, 18), (16, 18), (17, 18)):
    P, fl = apply_hungarian(P0, g, lo=lo, hi=hi)
    print(f"   {lo:2d}-{hi}: ba={ba(P):.4f} ({ba(P)-ba(P0):+.4f}) flips={fl}")

# how bad is a merged cluster? split-induced damage estimate
print("\nper-cluster purity of the best grouping (fraction of rows whose cluster is a single prompt):")
sizes = pd.Series(g)
pure = 0
for gid, idx in pd.Series(range(n)).groupby(sizes):
    if len(idx) < 12:
        continue
    pure += (pd.Series(truth[idx]).nunique() == 1) * len(idx)
print(f"   rows in 12-18 clusters belonging to a single true prompt: {pure}/{best[2]}")
