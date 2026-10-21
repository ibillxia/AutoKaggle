"""Multi-seed check of the 3-group cascade, which beat flat on the seed-42 holdout.

Groups come from the OOF-confusion clustering in hier_groups.py (seed 42):
  G1 gemini family | G2 gpt + qwen + kimi | G3 claude + glm
Five different 20% grouped holdouts quantify how much of the +0.007 is noise.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
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
G3 = {}
for m in VENDOR["google"]:
    G3[m] = "gemini"
for m in VENDOR["openai"] + ["qwen3.8-max", "kimi-k3"]:
    G3[m] = "gptish"
for m in VENDOR["anthropic"] + ["glm-5.3", "glm-5.3-flash"]:
    G3[m] = "claudeish"
C = 100.0


def cascade(Xtr, ytr, Xho, yho, assign):
    gtr = np.array([assign[m] for m in ytr])
    gs = sorted(set(assign.values()))
    gof = np.array([gs.index(assign[m]) for m in MODELS])
    l1 = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, gtr)
    o = [list(l1.classes_).index(g) for g in gs]
    P_g = l1.predict_proba(Xho)[:, o]
    l1acc = balanced_accuracy_score([assign[m] for m in yho], [gs[i] for i in P_g.argmax(1)])
    P_cond = np.zeros((len(Xho), len(MODELS)))
    for g in gs:
        cols = [MODELS.index(m) for m in MODELS if assign[m] == g]
        mask = gtr == g
        if len(cols) == 1 or mask.sum() < 40:
            P_cond[:, cols] = 1.0
            continue
        sub = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr[mask], ytr[mask])
        oo = [list(sub.classes_).index(m) for m in MODELS if assign[m] == g]
        P_cond[:, cols] = sub.predict_proba(Xho)[:, oo]
    keep = gof[None, :] == P_g.argmax(1)[:, None]
    hard = balanced_accuracy_score(yho, [MODELS[i] for i in np.argmax(np.where(keep, P_cond, 0), 1)])
    soft = P_g[:, gof] * P_cond
    soft /= np.clip(soft.sum(1, keepdims=True), 1e-12, None)
    return l1acc, hard, balanced_accuracy_score(yho, [MODELS[i] for i in soft.argmax(1)]), P_g


def main():
    tr = pd.read_csv("data/train.csv")
    d = np.load("emb_qwen4b_train.npz", allow_pickle=True)
    layers = sorted(int(k[1:]) for k in d.files if k.startswith("L"))
    X = np.concatenate([d[f"L{L}"] for L in layers], axis=1).astype(np.float32)
    assert (d["y"] == tr["model"].values).all()
    ym = tr["model"].values
    rows = []
    for seed in (42, 7, 2024, 13, 99):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    .split(tr, ym, groups=tr["prompt_id"]))
        sc = StandardScaler().fit(X[a])
        Xtr, Xho = sc.transform(X[a]).astype(np.float32), sc.transform(X[b]).astype(np.float32)
        ytr, yho = ym[a], ym[b]
        flat = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr)
        o = [list(flat.classes_).index(m) for m in MODELS]
        P = flat.predict_proba(Xho)[:, o]
        f = balanced_accuracy_score(yho, [MODELS[i] for i in P.argmax(1)])
        l1_3, hard_3, soft_3, P_g3 = cascade(Xtr, ytr, Xho, yho, G3)
        l1_v, hard_v, soft_v, P_gv = cascade(Xtr, ytr, Xho, yho,
                                             {m: v for v, ms in VENDOR.items() for m in ms})
        Lp = np.log(np.clip(P, 1e-9, 1))
        gof3 = np.array([sorted(set(G3.values())).index(G3[m]) for m in MODELS])
        geo3 = balanced_accuracy_score(yho, [MODELS[i] for i in np.argmax(Lp + 0.5 * np.log(np.clip(P_g3, 1e-9, 1))[:, gof3], 1)])
        print(f"seed {seed:5d}  flat={f:.4f}  k3: l1={l1_3:.3f} hard={hard_3:.4f} soft={soft_3:.4f} geo={geo3:.4f}"
              f"  | vendor: l1={l1_v:.3f} hard={hard_v:.4f}", flush=True)
        rows.append((f, hard_3, soft_3, geo3, hard_v))

    r = np.array(rows)
    names = ["flat", "k3_hard", "k3_soft", "k3_geo", "vendor_hard"]
    print("\n" + "  ".join(f"{n}={r[:, i].mean():.4f}±{r[:, i].std():.4f}" for i, n in enumerate(names)))
    print(f"k3_hard - flat = {(r[:, 1] - r[:, 0]).mean():+.4f} (per-seed: {np.round(r[:, 1] - r[:, 0], 4)})")


if __name__ == "__main__":
    main()
