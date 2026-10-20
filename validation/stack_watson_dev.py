"""Stacked (learned) ensemble over several watson members, evaluated with K-fold CV.

    python stack_watson_dev.py --runs xlmr-fb12-big-tta,xlmr-fb12-huge2-tta,xlmr-full-ld-tta --lang-feats 0

Why: plain weighted averaging of calibrated probabilities plateaus around 0.863 COMP-MIX.
A multinomial logistic regression over the members' calibrated log-probs can learn
per-class / per-language reweighting and usually buys another few thousandths.

CV is stratified on (language, label) so every fold keeps the instrument's balance.
The reported number is out-of-fold, i.e. an honest estimate of what the stacker does
on unseen rows.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path("E:/Bill/AI/AutoKaggle")
DEV = ROOT / "data/comp/contradictory-my-dear-watson"
CLASSES = ["entailment", "neutral", "contradiction"]
ABBR = {"english": "en", "arabic": "ar", "bulgarian": "bg", "german": "de", "greek": "el",
        "spanish": "es", "french": "fr", "hindi": "hi", "russian": "ru", "swahili": "sw",
        "thai": "th", "turkish": "tr", "urdu": "ur", "vietnamese": "vi", "chinese": "zh"}


def comp_mix():
    c = Counter()
    with (DEV / "test.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            c[ABBR[r["language"].strip().lower()]] += 1
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()}


def fit_T(p, y):
    """one-parameter temperature fit by grid search on NLL (same objective as fit_temperature)"""
    best = (1e9, 1.0)
    for t in np.arange(0.5, 3.01, 0.01):
        lp = np.log(np.clip(p, 1e-12, None)) / t
        lp = lp - lp.max(1, keepdims=True)
        e = np.exp(lp)
        e /= e.sum(1, keepdims=True)
        nll = -np.mean(np.log(np.clip(e[np.arange(len(y)), y], 1e-12, None)))
        if nll < best[0]:
            best = (nll, float(t))
    return best[1]


def load_run(run):
    path = ROOT / "validation/runs" / f"{run}-dev" / "rows.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    p = np.array([[r["p"][r["keys"].index(c)] for c in CLASSES] for r in rows], dtype=np.float64)
    y = np.array([r["label"] if isinstance(r["label"], int) else r["keys"].index(r["label"])
                  for r in rows], dtype=int)
    return p, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lang-feats", type=int, default=0)
    ap.add_argument("--per-lang", type=int, default=0)
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()

    runs = a.runs.split(",")
    meta = json.loads((DEV / "kev_dev.meta.json").read_text(encoding="utf-8"))
    langs = [m["lang"] for m in meta]
    W = comp_mix()

    mats, ys = [], None
    for r in runs:
        p, y = load_run(r)
        if ys is None:
            ys = y
        assert np.array_equal(ys, y), "label order mismatch"
        t = fit_T(p, y)
        lp = np.log(np.clip(p, 1e-12, None)) / t
        mats.append(lp)
        print("%-24s T=%.2f" % (r, t), flush=True)
    X = np.concatenate(mats, 1)
    y = ys
    n = len(y)

    lang_idx = {L: i for i, L in enumerate(sorted(set(langs)))}
    li = np.array([lang_idx[L] for L in langs])
    if a.lang_feats:
        oh = np.zeros((n, len(lang_idx)))
        oh[np.arange(n), li] = 1.0
        X = np.concatenate([X, oh], 1)
    print("features %d  rows %d  members %d" % (X.shape[1], n, len(runs)), flush=True)

    strat = np.array(["%s|%d" % (L, c) for L, c in zip(langs, y)])
    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    oof = np.zeros((n, 3))
    for k, (tr, te) in enumerate(skf.split(X, strat)):
        clf = LogisticRegression(C=a.C, max_iter=3000)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])
        print("  fold %d/%d done" % (k + 1, a.folds), flush=True)

    pred = oof.argmax(1)
    plain = float(np.mean(pred == y))
    by = Counter()
    tot = Counter()
    for L, ok in zip(langs, pred == y):
        by[L] += int(ok)
        tot[L] += 1
    cmix = sum(W[L] * by[L] / tot[L] for L in W if tot[L]) / sum(W[L] for L in W if tot[L])
    print("STACK oof acc=%.4f  COMP-MIX=%.4f" % (plain, cmix), flush=True)
    print("per-lang:", " ".join("%s=%.3f" % (L, by[L] / tot[L]) for L in sorted(by)), flush=True)


if __name__ == "__main__":
    main()
