# Three-way blend on the grouped holdout: centred TF-IDF, kev-0.8B, roberta-large.
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]

z = np.load("holdout_probs_tfidf.npz", allow_pickle=True)
P_tf = z["ymix"]
y = z["y"]

rows = [json.loads(l) for l in open("kev_preds.jsonl", encoding="utf-8")]
P_kev = np.array([[r["probs"][m] for m in MODELS] for r in rows])

r = np.load("E:/Bill/AI/AutoKaggle/runs/ntl-roberta-large/holdout_probs.npz", allow_pickle=True)
P_rob = r["probs"]
assert (np.array(MODELS)[r["y"]] == y).all(), "holdout row order mismatch"

def acc(P):
    return balanced_accuracy_score(y, [MODELS[i] for i in np.argmax(P, 1)])

print(f"tfidf   {acc(P_tf):.4f}\nkev     {acc(P_kev):.4f}\nroberta {acc(P_rob):.4f}")

L = {k: np.log(np.clip(P, 1e-9, 1)) for k, P in
     (("tf", P_tf), ("kev", P_kev), ("rob", P_rob))}
best = None
for w_kev in np.arange(0, 0.55, 0.05):
    for w_rob in np.arange(0, 0.55, 0.05):
        w_tf = 1 - w_kev - w_rob
        if w_tf <= 0:
            continue
        P = w_tf * L["tf"] + w_kev * L["kev"] + w_rob * L["rob"]
        s = acc(P)
        if best is None or s > best[1]:
            best = ((round(w_tf, 2), round(w_kev, 2), round(w_rob, 2)), s)
print("best logit blend weights (tf, kev, rob):", best[0], f"bal_acc={best[1]:.4f}")

# pairwise for reference
for a, b, name in (("tf", "kev", "tf+kev"), ("tf", "rob", "tf+rob")):
    bb = None
    for w in np.arange(0, 0.55, 0.05):
        s = acc((1 - w) * L[a] + w * L[b])
        if bb is None or s > bb[1]:
            bb = (round(1 - w, 2), round(w, 2), s)
    print(f"{name}: {bb}")
