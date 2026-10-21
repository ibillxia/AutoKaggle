"""Does the *choice* of level-1 grouping matter, and how many groups is best?

Level-1 groups are derived from the OOF confusion matrix (5-fold grouped CV on the 4B
embeddings): classes that the flat model confuses go into the same group.  For each k we
report hard cascade, soft hierarchy, geo reweighting and the oracle upper bound, plus the
vendor grouping as reference.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

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
    "alibaba": ["qwen3.8-max"], "zhipu": ["glm-5.3", "glm-5.3-flash"], "moonshot": ["kimi-k3"],
}
C = 100.0


def oof_probs(X, y, groups, n_splits=5):
    P = np.zeros((len(y), len(MODELS)))
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = LogisticRegression(C=C, max_iter=3000).fit(X[tr], y[tr])
        order = [list(clf.classes_).index(m) for m in MODELS]
        P[va] = clf.predict_proba(X[va])[:, order]
    return P


def group_metrics(Xtr, ytr, Xho, yho, assign, tag):
    """assign: dict model -> group id. Trains level-1 and level-2, returns all variants."""
    gtr = np.array([assign[m] for m in ytr])
    gs = sorted(set(assign.values()))
    gof = np.array([gs.index(assign[m]) for m in MODELS])

    l1 = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, gtr)
    order = [list(l1.classes_).index(g) for g in gs]
    P_g = l1.predict_proba(Xho)[:, order]
    l1acc = balanced_accuracy_score([assign[m] for m in yho], [gs[i] for i in P_g.argmax(1)])

    P_cond = np.zeros((len(Xho), len(MODELS)))
    for g in gs:
        cols = [MODELS.index(m) for m in MODELS if assign[m] == g]
        mask = gtr == g
        if len(cols) == 1 or mask.sum() < 40:
            P_cond[:, cols] = 1.0
        else:
            sub = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr[mask], ytr[mask])
            o = [list(sub.classes_).index(m) for m in MODELS if assign[m] == g]
            P_cond[:, cols] = sub.predict_proba(Xho)[:, o]

    multi = np.array([sum(1 for m2 in MODELS if assign[m2] == assign[m]) > 1 for m in MODELS])
    gtrue = np.array([gs.index(assign[m]) for m in yho])

    def acc(P, mask=None):
        pred = [MODELS[i] for i in np.argmax(P, 1)]
        if mask is None:
            return balanced_accuracy_score(yho, pred)
        cls = [m for m, k in zip(MODELS, mask) if k]
        return float(np.mean([np.mean([p == c for p, t in zip(pred, yho) if t == c]) for c in cls]))

    keep = gof[None, :] == P_g.argmax(1)[:, None]
    hard = acc(np.where(keep, P_cond, 0.0))
    soft = P_g[:, gof] * P_cond
    soft /= np.clip(soft.sum(1, keepdims=True), 1e-12, None)
    geo = None
    ko = gof[None, :] == gtrue[:, None]
    oracle = acc(np.where(ko, P_cond, 0.0))
    oracle16 = acc(np.where(ko, P_cond, 0.0), multi)
    sizes = [int((gtr == g).sum()) for g in gs]
    print(f"{tag:26s} k={len(gs):2d} l1={l1acc:.3f}  hard={hard:.4f}  soft={acc(soft):.4f}  oracle={oracle:.4f} (16cls {oracle16:.4f})  sizes={sizes}", flush=True)
    return dict(tag=tag, k=len(gs), l1=l1acc, hard=hard, soft=acc(soft), oracle=oracle, P_cond=P_cond, P_g=P_g)


def main():
    tr = pd.read_csv("data/train.csv")
    d = np.load("emb_qwen4b_train.npz", allow_pickle=True)
    layers = sorted(int(k[1:]) for k in d.files if k.startswith("L"))
    X = np.concatenate([d[f"L{L}"] for L in layers], axis=1).astype(np.float32)
    assert (d["y"] == tr["model"].values).all()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    a, b = next(gss.split(tr, tr["model"], groups=tr["prompt_id"]))
    sc = StandardScaler().fit(X[a])
    Xall = sc.transform(X).astype(np.float32)
    ytr, yho = tr["model"].values[a], tr["model"].values[b]

    print("computing OOF confusion on the training split ...", flush=True)
    P_oof = oof_probs(Xall[a], ytr, tr["prompt_id"].values[a])
    pred = [MODELS[i] for i in P_oof.argmax(1)]
    print(f"  OOF bal_acc = {balanced_accuracy_score(ytr, pred):.4f}")
    Cm = confusion_matrix(ytr, pred, labels=MODELS).astype(float)
    Cm /= np.clip(Cm.sum(1, keepdims=True), 1, None)
    S = (Cm + Cm.T) / 2
    np.fill_diagonal(S, 1.0)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")

    base = LogisticRegression(C=C, max_iter=3000).fit(Xall[a], ytr)
    order = [list(base.classes_).index(m) for m in MODELS]
    P_flat = base.predict_proba(Xall[b])[:, order]
    flat = balanced_accuracy_score(yho, [MODELS[i] for i in P_flat.argmax(1)])
    print(f"\nflat 18-way baseline: {flat:.4f}\n")

    results = []
    # vendor grouping as reference
    results.append(group_metrics(Xall[a], ytr, Xall[b], yho,
                                 {m: v for v, ms in VENDOR.items() for m in ms}, "vendor(prior)"))
    for k in (2, 3, 4, 5, 6, 7, 9, 12):
        lab = fcluster(Z, t=k, criterion="maxclust")
        assign = {m: int(l) for m, l in zip(MODELS, lab)}
        results.append(group_metrics(Xall[a], ytr, Xall[b], yho, assign, f"confusion-cluster k={k}"))
        print("      groups: " + " | ".join(
            "+".join(m for m in MODELS if assign[m] == g) for g in sorted(set(assign.values()))))

    np.savez("hier_group_results.npz", flat=flat,
             P_flat=P_flat, y=yho,
             **{f"cond_{i}": r["P_cond"] for i, r in enumerate(results)})
    print("\nsaved hier_group_results.npz")


if __name__ == "__main__":
    main()
