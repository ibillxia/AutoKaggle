"""Simulate decoding schemes on the official train layout.

The hidden test turned out to reward the centred twin strongly (0.520 -> 0.576), so the
remaining question is how to turn probabilities into labels once prompt groups are known.
Three candidates: plain argmax, Sinkhorn (uniform class prior), and a per-group Hungarian
assignment enforcing "each of the 18 models appears exactly once per prompt".

Absolute numbers mean nothing here -- the probabilities are synthetic. What matters is the
*difference* between decoders at the same simulated accuracy, and how that difference
changes with group quality.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer

RNG = np.random.default_rng(0)
tr = pd.read_csv("data/ntl/train.csv")
texts = tr.text.fillna("").astype(str).tolist()
truth_ids = np.array(tr.prompt_id.astype(str).tolist(), dtype=object)
LABELS = sorted(tr.model.unique().tolist())
K = len(LABELS)
y = tr.model.map({l: i for i, l in enumerate(LABELS)}).values
n = len(tr)


def bal_acc(pred, yy=None):
    yy = y if yy is None else yy
    rec = [np.mean(pred[yy == k] == k) for k in range(K) if (yy == k).any()]
    return float(np.mean(rec))


Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
D = np.clip(1 - (Xt @ Xt.T).toarray(), 0, None)
np.fill_diagonal(D, 0)
Za = linkage(squareform(D, checks=False), method="average")
Zw = linkage(squareform(D, checks=False), method="ward")
g_cur = fcluster(Za, 0.85, criterion="distance")      # what the live notebook uses
g_ward = fcluster(Zw, len(np.unique(truth_ids)), criterion="maxclust")
g_oracle = pd.factorize(truth_ids)[0] + 1

for name, g in (("current avg/0.85", g_cur), ("ward k=#prompts", g_ward), ("oracle", g_oracle)):
    s = pd.Series(g)
    sz = s.map(s.value_counts())
    print(f"{name:<18} groups={s.nunique():4d} rows in 18-groups={int((sz == 18).sum()):4d} "
          f"rows in 2-18={int(((sz >= 2) & (sz <= 18)).sum()):4d}")

# ---- correlated confusion: errors stay inside model families, like the real task ----
FAMILY = {
    "gpt": [i for i, l in enumerate(LABELS) if l.startswith("gpt")],
    "gemini": [i for i, l in enumerate(LABELS) if l.startswith("gemini")],
    "claude": [i for i, l in enumerate(LABELS) if l.startswith("claude")],
    "other": [i for i, l in enumerate(LABELS) if l.split("-")[0].split(".")[0] in ("qwen3", "glm", "kimi")],
}
fam_of = np.zeros(K, int)
for f, idx in enumerate(FAMILY.values()):
    fam_of[idx] = f

CONF = np.full((K, K), 0.12)                 # cross-family leakage
for idx in FAMILY.values():
    for a in idx:
        for b in idx:
            CONF[a, b] = 1.0 if a != b else 0.0
CONF /= CONF.sum(1, keepdims=True)           # P(pred class | true class), errors inside family


def make_probs(scale):
    """Synthetic probabilities whose argmax accuracy tracks `scale`."""
    z = RNG.gumbel(size=(n, K)) * 1.0
    for i in range(n):
        r = RNG.random()
        if r < 0.0:
            pass
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


def group_hungarian(P, g, exact=18, min_size=12):
    """Within each group, pick an assignment where every class is used exactly once."""
    out = P.argmax(1).copy()
    s = pd.Series(g)
    sizes = s.map(s.value_counts())
    idx_by_g = {}
    for i, gg in enumerate(g):
        idx_by_g.setdefault(gg, []).append(i)
    for gg, rows in idx_by_g.items():
        if not (min_size <= len(rows) <= exact):
            continue
        cost = -np.log(np.clip(P[np.ix_(rows, np.arange(K))], 1e-12, 1))
        # pad/truncate classes to the number of rows: capacity = ceil(len/K) per class
        cap = int(np.ceil(len(rows) / K))
        big = np.repeat(cost, cap, axis=1)
        r, c = linear_sum_assignment(big)
        out[np.array(rows)[r]] = c // cap
    return out


print("\nsimulated accuracy sweep (bal acc; synthetic probs, compare decoders at equal input)")
print(f"{'scale':>6} {'argmax':>8} {'sinkhorn':>9} {'grp(H)':>8} {'grp+H':>8} | "
      f"{'argmax_w':>9} {'grp_w':>8} {'grpO':>7}")
for scale in (1.0, 2.0, 3.0):
    P = softmax(make_probs(scale))
    a = bal_acc(P.argmax(1))
    sk = bal_acc(sinkhorn(P).argmax(1))
    gh = bal_acc(group_hungarian(P, g_cur))
    gh_sk = bal_acc(group_hungarian(sinkhorn(P), g_cur))
    aw = bal_acc(group_hungarian(P, g_ward, min_size=99))      # argmax only, ward grouping
    gw = bal_acc(group_hungarian(P, g_ward))
    go = bal_acc(group_hungarian(P, g_oracle))
    print(f"{scale:>6.1f} {a:>8.4f} {sk:>9.4f} {gh:>8.4f} {gh_sk:>8.4f} | "
          f"{aw:>9.4f} {gw:>8.4f} {go:>7.4f}")
