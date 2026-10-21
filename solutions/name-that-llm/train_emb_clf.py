# Train and evaluate a linear classifier on Qwen3.5-4B layer embeddings (official train + extra corpus).
# Holdout = 30 official prompts never seen in training; extra corpus comes from different prompts entirely.
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]

def load(npz="emb_qwen4b.npz"):
    d = np.load(npz, allow_pickle=True)
    layers = sorted(int(k[1:]) for k in d.files if k.startswith("L"))
    return {L: d[f"L{L}"] for L in layers}, d["y"], d["src"]

def build_X(emb, layers, scale=True):
    X = np.concatenate([emb[L] for L in layers], axis=1).astype(np.float32)
    return StandardScaler().fit(X).transform(X) if scale else X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="emb_qwen4b.npz")
    ap.add_argument("--use_extra", type=int, default=1)
    ap.add_argument("--cap", type=int, default=0, help="cap extra rows per class (0 = no cap)")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--out_probs", default="")
    a = ap.parse_args()

    emb, y, src = load(a.npz)
    layers = sorted(emb)
    print("layers:", layers, "rows:", len(y), pd.Series(src).value_counts().to_dict())

    tr = pd.read_csv("data/train.csv")
    assert len(tr) == (src == "train").sum()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, ho_idx = next(gss.split(tr, tr["model"], groups=tr["prompt_id"]))
    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[tr_idx] = True
    hold_mask = np.zeros(len(y), dtype=bool)
    hold_mask[ho_idx] = True

    fit_mask = train_mask.copy()
    if a.use_extra:
        fit_mask |= (src == "extra")
    if a.cap:
        keep = np.zeros(len(y), dtype=bool)
        keep[fit_mask & (src == "train")] = True
        dfm = pd.DataFrame({"i": np.arange(len(y)), "y": y, "src": src})[fit_mask]
        for cls, g in dfm.groupby("y"):
            ex = g[g.src == "extra"]
            if len(ex) > a.cap:
                ex = ex.sample(a.cap, random_state=0)
            keep[ex["i"].values] = True
        fit_mask = keep
    print("fit rows:", fit_mask.sum(), "holdout rows:", hold_mask.sum())

    def evaluate(layer_sets, C):
        X = build_X(emb, layer_sets)
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", n_jobs=-1)
        clf.fit(X[fit_mask], y[fit_mask])
        pred = clf.predict(X[hold_mask])
        acc = balanced_accuracy_score(y[hold_mask], pred)
        order = [list(clf.classes_).index(m) for m in MODELS]
        return acc, clf.predict_proba(X[hold_mask])[:, order], clf

    results = {}
    for layers_sub in ([3], [1, 3, 6], [3, 6, 9, 16, 24, 32], layers):
        acc, P, clf = evaluate(layers_sub, a.C)
        tag = "+".join(map(str, layers_sub))
        results[tag] = (acc, P)
        print(f"layers {tag:22s} C={a.C} bal_acc={acc:.4f}", flush=True)

    best_tag = max(results, key=lambda k: results[k][0])
    print("best:", best_tag, round(results[best_tag][0], 4))
    np.savez("emb_holdout_probs.npz", probs=results[best_tag][1], y=y[hold_mask],
             classes=np.array(MODELS), hold_idx=np.where(hold_mask)[0], tag=best_tag)

if __name__ == "__main__":
    main()
