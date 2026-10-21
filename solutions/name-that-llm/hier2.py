"""Round 2 of the hierarchy study: fair oracle, prior replacement, trained-free level-2,
and blending the two feature spaces before/after the hierarchy is applied.

Reads hier_probs_{tfidf,emb}.npz written by hier_clf.py.
"""
import itertools

import numpy as np
from sklearn.metrics import balanced_accuracy_score

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
    "alibaba": ["qwen3.8-max"],
    "zhipu": ["glm-5.3", "glm-5.3-flash"],
    "moonshot": ["kimi-k3"],
}
VENDORS = sorted(VENDOR)
VMAP = {m: v for v, ms in VENDOR.items() for m in ms}
VOF = np.array([VENDORS.index(VMAP[m]) for m in MODELS])
MULTI = np.array([len(VENDOR[VMAP[m]]) > 1 for m in MODELS])   # model columns with a real level-2


def acc(truth, P, mask=None):
    """Balanced accuracy over all 18 classes, or only over `mask`-ed classes (truth rows per class)."""
    pred = [MODELS[i] for i in np.argmax(P, 1)]
    if mask is None:
        return balanced_accuracy_score(truth, pred)
    cls = [m for m, k in zip(MODELS, mask) if k]
    per = [np.mean([p == c for p, t in zip(pred, truth) if t == c]) for c in cls]
    return float(np.mean(per))


def softmax(L):
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def norm(P):
    return P / np.clip(P.sum(1, keepdims=True), 1e-12, None)


def load(feat):
    d = np.load(f"hier_probs_{feat}.npz", allow_pickle=True)
    return d["P_flat"], d["P_v"], d["P_cond"], d["y"]


def evaluate(tag, P_flat, P_v, P_cond, y, extra=""):
    Lp = np.log(np.clip(P_flat, 1e-9, 1))
    Lv_all = np.log(np.clip(P_v, 1e-9, 1))
    vtrue = np.array([VENDORS.index(VMAP[m]) for m in y])
    out = {}
    out["flat"] = acc(y, P_flat)
    out["flat_16"] = acc(y, P_flat, MULTI)
    ko = VOF[None, :] == vtrue[:, None]
    out["oracle16"] = acc(y, np.where(ko, P_cond, 0.0), MULTI)
    # prior replacement: swap the flat model's own vendor marginal for the level-1 posterior
    P_marg = np.zeros_like(P_flat)
    for k in range(len(VENDORS)):
        cols = VOF == k
        P_marg[:, cols] = P_flat[:, cols].sum(1, keepdims=True)
    LPr = Lp + Lv_all[:, VOF] - np.log(np.clip(P_marg, 1e-9, 1))
    out["prior_replace"] = acc(y, norm(np.exp(LPr - LPr.max(1, keepdims=True))))
    # free level-2: renormalise flat probs inside each vendor, no extra training
    P_free = np.zeros_like(P_flat)
    for k in range(len(VENDORS)):
        cols = VOF == k
        P_free[:, cols] = norm(P_flat[:, cols])
    out["free_cond"] = acc(y, P_free)
    print(f"[{tag}] " + "  ".join(f"{k}={v:.4f}" for k, v in out.items()), extra)
    return out


def main():
    Pf_t, Pv_t, Pc_t, y = load("tfidf")
    Pf_e, Pv_e, Pc_e, y2 = load("emb")
    assert (y == y2).all()

    print("== single feature spaces ==")
    evaluate("tfidf", Pf_t, Pv_t, Pc_t, y)
    evaluate("emb  ", Pf_e, Pv_e, Pc_e, y)
    print("   (flat_16 / oracle16 restrict the macro average to the 16 classes that have a real level-2)")

    # --- hierarchy with soft level-1 ---
    for tag, Pf, Pv, Pc in (("tfidf", Pf_t, Pv_t, Pc_t), ("emb", Pf_e, Pv_e, Pc_e)):
        Ps = norm(Pv[:, VOF] * Pc)
        print(f"[{tag}] hierarchy soft = {acc(y, Ps):.4f}")

    print("\n== blended feature spaces (log-average of probabilities, then hierarchy) ==")
    best = {}
    for w in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        Pfl = norm(softmax((1 - w) * np.log(np.clip(Pf_t, 1e-9, 1)) + w * np.log(np.clip(Pf_e, 1e-9, 1))))
        Pv = norm(softmax((1 - w) * np.log(np.clip(Pv_t, 1e-9, 1)) + w * np.log(np.clip(Pv_e, 1e-9, 1))))
        Pc = norm((1 - w) * Pc_t + w * Pc_e)
        flat = acc(y, Pfl)
        hier = acc(y, norm(Pv[:, VOF] * Pc))
        geo = acc(y, norm(np.exp(np.log(np.clip(Pfl, 1e-9, 1)) + 0.5 * np.log(np.clip(Pv, 1e-9, 1))[:, VOF])))
        print(f"  w_emb={w:.1f}  flat={flat:.4f}  hier={hier:.4f}  flat+geo0.5={geo:.4f}")
        best[w] = (flat, hier, geo)

    # vendor level diagnostic on the blended space
    w = max(best, key=lambda k: best[k][0])
    print(f"\nbest flat weight w_emb={w}")


if __name__ == "__main__":
    main()
