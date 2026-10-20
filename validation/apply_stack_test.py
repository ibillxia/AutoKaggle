"""Fit the stacked ensemble on the dev instrument and write a submission for test.csv.

    python apply_stack_test.py \
        --member xlmr-fb12-pool:submissions/xlmr_pool_test_probs.jsonl \
        --member xlmr-fb12-big:submissions/xlmr_big_test_probs.jsonl \
        ... --out submissions/stack.csv

The recipe is the one validated by stack_watson_dev.py: per-member temperature
calibration, then a multinomial logistic regression over the calibrated log-probs
plus a one-hot language indicator.  Language is known for every test row
(test.csv carries a `language` column), so the same feature is available at
inference time.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path("E:/Bill/AI/AutoKaggle")
DEV = ROOT / "data/comp/contradictory-my-dear-watson"
CLASSES = ["entailment", "neutral", "contradiction"]
ABBR = {"english": "en", "arabic": "ar", "bulgarian": "bg", "german": "de", "greek": "el",
        "spanish": "es", "french": "fr", "hindi": "hi", "russian": "ru", "swahili": "sw",
        "thai": "th", "turkish": "tr", "urdu": "ur", "vietnamese": "vi", "chinese": "zh"}


def fit_T(p, y):
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


def load_dev(run):
    rows = json.loads((ROOT / "validation/runs" / f"{run}-dev" / "rows.json").read_text(encoding="utf-8"))
    p = np.array([[r["p"][r["keys"].index(c)] for c in CLASSES] for r in rows], dtype=np.float64)
    y = np.array([r["label"] if isinstance(r["label"], int) else r["keys"].index(r["label"])
                  for r in rows], dtype=int)
    return p, y


def load_test(path):
    ids, p = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            keys = r.get("keys", CLASSES)
            ids.append(r["id"])
            p.append([r["p"][keys.index(c)] for c in CLASSES])
    return ids, np.array(p, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True, help="run_name:test_probs.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()

    meta = json.loads((DEV / "kev_dev.meta.json").read_text(encoding="utf-8"))
    dev_langs = [m["lang"] for m in meta]
    lang_idx = {L: i for i, L in enumerate(sorted(set(dev_langs)))}

    tids, tlangs = [], []
    with (DEV / "test.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            tids.append(r["id"])
            tlangs.append(ABBR[r["language"].strip().lower()])
    print("test rows %d" % len(tids), flush=True)

    Xd_parts, Xt_parts = [], []
    y = None
    for spec in a.member:
        run, path = spec.split(":", 1)
        pd_, yd = load_dev(run)
        if y is None:
            y = yd
        else:
            assert np.array_equal(y, yd)
        t = fit_T(pd_, yd)
        tid_list, pt = load_test(path)
        assert tid_list == tids, "id order differs from test.csv (%s)" % run
        ld = np.log(np.clip(pd_, 1e-12, None)) / t
        lt = np.log(np.clip(pt, 1e-12, None)) / t
        Xd_parts.append(ld)
        Xt_parts.append(lt)
        print("  %-22s T=%.2f" % (run, t), flush=True)

    def onehot(langs):
        m = np.zeros((len(langs), len(lang_idx)))
        for i, L in enumerate(langs):
            m[i, lang_idx[L]] = 1.0
        return m

    Xd = np.concatenate(Xd_parts + [onehot(dev_langs)], 1)
    Xt = np.concatenate(Xt_parts + [onehot(tlangs)], 1)

    clf = LogisticRegression(C=a.C, max_iter=5000)
    clf.fit(Xd, y)
    prob = clf.predict_proba(Xt)
    pred = prob.argmax(1)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "prediction"])
        for i, p in zip(tids, pred):
            w.writerow([i, int(p)])
    dist = {c: int((pred == k).sum()) for k, c in enumerate(CLASSES)}
    print("wrote %s  %s" % (out, dist), flush=True)


if __name__ == "__main__":
    main()
