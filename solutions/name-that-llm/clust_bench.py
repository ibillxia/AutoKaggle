"""Benchmark prompt-group recovery on the official train split (ground truth = prompt_id).

The centred twin only pays off where the recovered group is the real prompt, so cluster
quality is the bottleneck. Metrics: pairwise F1 over same/different pairs (what centring
actually consumes), plus coverage of rows living in an accepted cluster.
"""
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer

tr = pd.read_csv("data/ntl/train.csv")
texts = tr.text.fillna("").astype(str).tolist()
truth = np.array(tr.prompt_id.astype(str).tolist(), dtype=object)
n = len(tr)
true_sizes = pd.Series(truth).value_counts()
per_prompt = int(round(true_sizes.median()))
k_true = tr.prompt_id.nunique()
print(f"n={n} prompts={k_true} median size={per_prompt}")


def pair_f1(pred, idx=None):
    """F1 over pairs, restricted to rows that got a cluster (others count as singletons)."""
    if idx is None:
        idx = np.arange(len(pred))
    p = np.asarray(pred)[idx]
    t = truth[idx]
    same_p = p[:, None] == p[None, :]
    same_t = t[:, None] == t[None, :]
    np.fill_diagonal(same_p, False)
    np.fill_diagonal(same_t, False)
    tp = (same_p & same_t).sum()
    fp = (same_p & ~same_t).sum()
    fn = (~same_p & same_t).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec


def coverage(pred, lo=2, hi=18):
    s = pd.Series(pred)
    sz = s.map(s.value_counts())
    return float(((sz >= lo) & (sz <= hi)).mean())


def report(name, pred, keep=None):
    f1, p, r = pair_f1(pred)
    print(f"{name:<44} F1={f1:.4f} P={p:.4f} R={r:.4f} coverage(2-18)={coverage(pred):.3f}")
    return f1


Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
D = np.clip(1 - (Xt @ Xt.T).toarray(), 0, None)
np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), method="average")
print("linkage done")

# A: exactly what fork592 does
for th in (0.75, 0.80, 0.85, 0.90, 0.95):
    A = fcluster(Z, th, criterion="distance")
    report(f"A avg-linkage thresh={th}", A)

# B: cut the same tree to the known k (test rows per prompt is not known a priori,
#    but n/18 is a serviceable estimate when the layout is 18 per prompt)
for k in (k_true - 20, k_true, k_true + 20, k_true + 60):
    B = fcluster(Z, k, criterion="maxclust")
    report(f"B avg-linkage k={k}", B)

# C: correlation/ward variants of the same distance matrix
for meth in ("complete", "ward", "single"):
    Zm = linkage(squareform(D, checks=False), method=meth)
    C = fcluster(Zm, k_true, criterion="maxclust")
    report(f"C {meth} k={k_true}", C)


def rebalance(pred, target):
    """Grow/shrink clusters to `target` rows: surplus rows are re-seated by Hungarian
    cost (distance to every cluster centroid), so every cluster ends up exactly `target`."""
    pred = np.asarray(pred).copy()
    labs = np.unique(pred)
    cents = np.vstack([Xt[pred == l].mean(0) for l in labs])
    cents = np.asarray(cents).reshape(len(labs), -1)
    Cn = cents / np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-9)
    Xn = Xt.copy()
    sim = np.asarray(Xn @ Cn.T)          # (n, k) cosine similarity to each centroid
    cost = 1.0 - sim
    nk = len(labs)
    slots = int(round(n / nk))
    # capacity-limited assignment: repeat each cluster `slots` times as columns
    big = np.repeat(cost, slots, axis=1)
    r, c = linear_sum_assignment(big)
    out = np.empty(len(pred), dtype=int)
    out[r] = labs[(c // slots)]
    return out


best = fcluster(Z, 0.85, criterion="distance")
report("A chosen for rebalance (0.85)", best)
t = time.time()
for lab in (0.85, 0.90):
    base = fcluster(Z, lab, criterion="distance")
    # drop singletons? rebalance needs every row assigned anyway
    R = rebalance(base, per_prompt)
    report(f"D rebalance(thresh={lab}) -> size {per_prompt}", R)
print(f"rebalance took {time.time()-t:.0f}s")
