"""Second decode simulation: does Sinkhorn help or hurt once the per-group matching is applied?

sim_decode.py compared plain argmax / Sinkhorn / group-matching separately, so the exact
combination that ships (Sinkhorn -> ward groups -> Hungarian) was never measured. Sinkhorn
distorts the rows *before* the matching sees them, and the matching already forces a uniform
class marginal inside every group, so it may be redundant or harmful there.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer

RNG = np.random.default_rng(0)
tr = pd.read_csv("../../data/ntl/train.csv")
texts = tr.text.fillna("").astype(str).tolist()
truth_ids = np.array(tr.prompt_id.astype(str).tolist(), dtype=object)
LABELS = sorted(tr.model.unique().tolist())
K = len(LABELS)
y = tr.model.map({l: i for i, l in enumerate(LABELS)}).values
n = len(tr)


def bal_acc(pred, yy=None):
    yy = y if yy is None else yy
    return float(np.mean([np.mean(pred[yy == k] == k) for k in range(K) if (yy == k).any()]))


Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
D = np.clip(1 - (Xt @ Xt.T).toarray(), 0, None)
np.fill_diagonal(D, 0)
Dc = squareform(D, checks=False)
g_ward = fcluster(linkage(Dc, method="ward"), len(np.unique(truth_ids)), criterion="maxclust")
g_oracle = pd.factorize(truth_ids)[0] + 1

FAMILY = {
    "gpt": [i for i, l in enumerate(LABELS) if l.startswith("gpt")],
    "gemini": [i for i, l in enumerate(LABELS) if l.startswith("gemini")],
    "claude": [i for i, l in enumerate(LABELS) if l.startswith("claude")],
    "other": [i for i, l in enumerate(LABELS) if l.split("-")[0].split(".")[0] in ("qwen3", "glm", "kimi")],
}
CONF = np.full((K, K), 0.12)
for idx in FAMILY.values():
    for a in idx:
        for b in idx:
            CONF[a, b] = 1.0 if a != b else 0.0
CONF /= CONF.sum(1, keepdims=True)


def make_probs(scale):
    z = RNG.gumbel(size=(n, K))
    for i in range(n):
        z[i, y[i]] += scale
    return z


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def sinkhorn(P, iters=100):
    P = P.copy()
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        P *= target / P.sum(0)
        P /= P.sum(1, keepdims=True)
    return P


def group_hungarian(P, g, min_size=12, max_size=18):
    out = P.argmax(1).copy()
    idx_by_g = {}
    for i, gg in enumerate(g):
        idx_by_g.setdefault(gg, []).append(i)
    for rows in idx_by_g.values():
        if not (min_size <= len(rows) <= max_size):
            continue
        cost = -np.log(np.clip(P[np.ix_(rows, np.arange(K))], 1e-12, 1))
        r, c = linear_sum_assignment(cost)
        out[np.array(rows)[r]] = c
    return out


print(f"{'scale':>6} {'argmax':>8} {'sink':>8} | {'wardH':>8} {'sink+wardH':>11} | "
      f"{'oracleH':>8} {'sink+oracleH':>13}")
for scale in (1.0, 2.0, 3.0):
    P = softmax(make_probs(scale))
    Pb = sinkhorn(P)
    print(f"{scale:>6.1f} {bal_acc(P.argmax(1)):>8.4f} {bal_acc(Pb.argmax(1)):>8.4f} | "
          f"{bal_acc(group_hungarian(P, g_ward)):>8.4f} {bal_acc(group_hungarian(Pb, g_ward)):>11.4f} | "
          f"{bal_acc(group_hungarian(P, g_oracle)):>8.4f} {bal_acc(group_hungarian(Pb, g_oracle)):>13.4f}")
