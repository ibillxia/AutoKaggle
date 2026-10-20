"""Fit a global mixture over members (per-member T fixed from dev) by minimizing dev NLL.

    python fit_mix_weights.py --runs a,b,c --out validation/mix7_weights.json

7 free weights, no language features, no per-class terms: far less dev-overfit than the
27-parameter language-aware logistic stack, so its dev gains should transfer to the LB better.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("E:/Bill/AI/AutoKaggle")
DEV = ROOT / "data/comp/contradictory-my-dear-watson"
CLASSES = ["entailment", "neutral", "contradiction"]


def load_run(run):
    rows = json.loads((ROOT / "validation/runs" / f"{run}-dev" / "rows.json").read_text(encoding="utf-8"))
    p = np.array([[r["p"][r["keys"].index(c)] for c in CLASSES] for r in rows], dtype=np.float64)
    y = np.array([r["label"] if isinstance(r["label"], int) else r["keys"].index(r["label"])
                  for r in rows], dtype=int)
    return p, y


def fit_T(p, y):
    best = (1e9, 1.0)
    for t in np.arange(0.5, 3.01, 0.01):
        lp = np.log(np.clip(p, 1e-12, None)) / t
        lp -= lp.max(1, keepdims=True)
        e = np.exp(lp)
        e /= e.sum(1, keepdims=True)
        nll = -np.mean(np.log(np.clip(e[np.arange(len(y)), y], 1e-12, None)))
        if nll < best[0]:
            best = (nll, float(t))
    return best[1]


def mix_nll(L, y, w):
    s = np.tensordot(w, L, axes=1)                # (n, 3)
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    e /= e.sum(1, keepdims=True)
    return -np.mean(np.log(np.clip(e[np.arange(len(y)), y], 1e-12, None)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    runs = a.runs.split(",")
    mats, y = [], None
    ts = []
    for r in runs:
        p, yy = load_run(r)
        if y is None:
            y = yy
        else:
            assert np.array_equal(y, yy)
        t = fit_T(p, y)
        ts.append(t)
        lp = np.log(np.clip(p, 1e-12, None)) / t
        mats.append(lp - lp.mean(0, keepdims=True))
        print("%-24s T=%.2f" % (r, t), flush=True)
    L = np.stack(mats, 0)                          # (m, n, 3)
    m = len(runs)
    n = len(y)

    rng = np.random.default_rng(a.seed)
    w = np.full(m, 1.0 / m)
    best_w, best = w.copy(), mix_nll(L, y, w)
    scale = 0.05
    for it in range(a.iters):
        cand = np.clip(w + rng.normal(0, scale, m), 1e-4, None)
        cand /= cand.sum()
        v = mix_nll(L, y, cand)
        if v < best:
            best, best_w = v, cand
            w = cand
        if (it + 1) % 500 == 0:
            scale = max(scale * 0.7, 0.002)
    pred = np.tensordot(best_w, L, axes=1).argmax(1)
    acc = float(np.mean(pred == y))
    print("best NLL %.4f  acc %.4f" % (best, acc), flush=True)
    for r, t, wi in zip(runs, ts, best_w):
        print("  %-24s T=%.2f w=%.4f" % (r, t, wi), flush=True)
    Path(a.out).write_text(json.dumps(
        {"runs": runs, "T": ts, "weights": [float(x) for x in best_w]}, indent=2), encoding="utf-8")
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
