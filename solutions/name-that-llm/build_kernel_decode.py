"""Build ntl-decode.ipynb: ens members + ward grouping + one-model-per-prompt assignment.

Two findings drive this:
1. The public notebook's average-linkage grouping (thresh 0.85) recovers the true prompt
   grouping at pairwise F1 0.877; ward linkage cut to n/18 groups reaches 0.985, essentially
   the oracle ceiling. Grouping quality is what the centred twin consumes.
2. Every prompt carries each of the 18 models exactly once (train: 150 prompts x 18). So
   within a recovered group, the labels are a perfect matching, not 18 independent argmaxes.
   A per-group Hungarian assignment on -log P enforces that. Simulated on the train layout
   with realistic correlated confusion, it moves 0.548 -> 0.603 with the old grouping and
   0.628 with ward grouping (oracle grouping: 0.631).

Writes submission.parquet (PICK=hung) plus an ablation file, all on ward groups:
  submission_hung.parquet   twin + Sinkhorn + per-group Hungarian
  submission_sink.parquet   twin + Sinkhorn, plain argmax
  submission_argmax.parquet twin only, plain argmax
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "decode")
os.makedirs(OUT_DIR, exist_ok=True)

SRC_NB = os.path.join(HERE, "ens", "ntl-ens.ipynb")   # already carries the 4 members + base pins
DST_NB = os.path.join(OUT_DIR, "ntl-decode.ipynb")

OLD_CLUSTER = '''Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(test.text.tolist())
D = np.clip(1 - (Xt @ Xt.T).toarray(), 0, None)
np.fill_diagonal(D, 0)
groups = fcluster(linkage(squareform(D, checks=False), method="average"), TWIN["thresh"], criterion="distance") \\
    if len(test) >= 2 else np.ones(len(test), dtype=int)
size = np.bincount(groups)[groups]
twin_rows = (size >= TWIN["min_size"]) & (size <= TWIN["max_size"])'''

NEW_CLUSTER = '''Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(test.text.tolist())
D = np.clip(1 - (Xt @ Xt.T).toarray(), 0, None)
np.fill_diagonal(D, 0)
Dc = squareform(D, checks=False)
del D
gc.collect()
n_t = len(test)
# The held-out layout is 18 passages per prompt (train: 150 prompts x 18, one per model), so
# n/18 is the number of prompts. ward on the cosine distance recovers them far better than
# average linkage at a fixed threshold (pairwise F1 0.985 vs 0.877 on train).
k_est = max(2, int(round(n_t / 18)))
if n_t >= 6:
    groups = fcluster(linkage(Dc, method="ward"), k_est, criterion="maxclust")
else:
    groups = np.ones(n_t, dtype=int)
    k_est = 1
del Dc
size = np.bincount(groups)[groups]
twin_rows = (size >= TWIN["min_size"]) & (size <= TWIN["max_size"])
print(f"ward groups: k={len(np.unique(groups))} (n/18={k_est}), twin rows {int(twin_rows.sum())}/{n_t}", flush=True)'''

FINAL_CELL = r'''
from scipy.optimize import linear_sum_assignment


def sinkhorn(P):
    P = P.copy()
    if len(P) < 50:          # the 3-row public test.csv: balancing is meaningless
        return P
    target = np.full(K, len(P) / K)
    for _ in range(100):
        P *= target / P.sum(0)
        P /= P.sum(1, keepdims=True)
    return P


def group_assignment(P, groups, min_size=12, max_size=18):
    """Each recovered prompt holds every model exactly once -> solve a matching, not argmax.
    min_size=12 is what sim_decode.py validated (ward groups of 12-18 rows)."""
    out = P.argmax(1).copy()
    fixed = 0
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        if not (min_size <= len(rows) <= max_size) or len(rows) > K:
            continue
        cost = -np.log(np.clip(P[np.ix_(rows, np.arange(K))], 1e-12, 1))
        r, c = linear_sum_assignment(cost)      # capacity 1 per class == one model per prompt
        out[rows[r]] = c
        fixed += len(rows)
    print(f"assignment applied to {fixed}/{len(P)} rows", flush=True)
    return out


P_twin = softmax(LOGIT_BOTH)
VARIANTS = {
    "submission_hung.parquet":   group_assignment(sinkhorn(P_twin), groups),
    "submission_sink.parquet":   sinkhorn(P_twin).argmax(1),
    "submission_argmax.parquet": P_twin.argmax(1),
}
for fname, pred in VARIANTS.items():
    pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[pred]}).to_parquet(fname, index=False)
    print(fname, "->", np.array(LABELS)[pred].tolist()[:3], flush=True)

PICK = "submission_hung.parquet"
pred = VARIANTS[PICK]
pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[pred]}).to_parquet("submission.parquet", index=False)
print(f"picked {PICK} -> submission.parquet")
print(f"done in {time.time() - T0:.0f}s")
'''


def main():
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    hits = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if OLD_CLUSTER in src:
            nb["cells"][i]["source"] = src.replace(OLD_CLUSTER, NEW_CLUSTER).splitlines(keepends=True)
            hits += 1
        elif "VARIANTS = {" in src:
            nb["cells"][i]["source"] = FINAL_CELL.strip("\n").splitlines(keepends=True)
            hits += 1
    assert hits == 2, f"expected 2 patched cells, got {hits}"
    json.dump(nb, open(DST_NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written", DST_NB)

    meta = json.load(open(os.path.join(HERE, "ens", "kernel-metadata.json"), encoding="utf-8"))
    meta.update({"id": "ibillxia/ntl-decode", "title": "ntl decode", "code_file": "ntl-decode.ipynb"})
    json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    print("written metadata")


if __name__ == "__main__":
    main()
