"""Classical baseline on the same 400 rows: TF-IDF (word 1-2 gram) + logistic regression.

Fitted on the task's own training split, which is the honest "what a Kaggle entrant would submit"
reference point for these datasets - not a zero-shot comparison, but the number a leaderboard
would show. Reports the same metric set as run_eval.py so the two tables line up.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np

HERE = Path(__file__).resolve().parent


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total, out = len(conf), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            out += m.sum() / total * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def load_test(task):
    rows = []
    with (HERE / "data" / f"{task}_test.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q = next(iter(r["questions"].values()))
            keys = list(q["criteria"]) if q["type"] == "choice" else ["false", "true"]
            label = q["label"]
            y = keys.index(label) if q["type"] == "choice" else int(label)
            rows.append((r["state"], y, keys))
    return rows


def load_train(task):
    from datasets import load_dataset
    if task == "disaster":
        # complement of the stratified holdout built by prep_data.py, same record shape
        X, Y = [], []
        with (HERE / "data" / "disaster_train.jsonl").open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                q = next(iter(r["questions"].values()))
                X.append(r["state"]); Y.append(int(q["label"]))
        return X, Y, ["false", "true"]
    if task == "imdb":
        ds = load_dataset("stanfordnlp/imdb", split="train")
        return list(ds["text"]), [int(y) for y in ds["label"]], ["false", "true"]
    if task == "sms_spam":
        # single-split dataset: train on everything that is not one of the 200 sampled rows
        ds = load_dataset("ucirvine/sms_spam", split="train")
        held = {s for s, _, _ in load_test("sms_spam")}
        X, Y = [], []
        for text, y in zip(ds["sms"], ds["label"]):
            if text.strip() in held:
                continue
            X.append(text.strip()); Y.append(int(y))
        return X, Y, ["false", "true"]
    ds = load_dataset("fancyzhx/ag_news", split="train")
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    return list(ds["text"]), [int(y) for y in ds["label"]], labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["imdb", "ag_news", "sms_spam", "disaster", "nli", "xnli"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    Xtr, ytr, labels = load_train(a.task)
    test = load_test(a.task)
    Xte, yte, _ = zip(*test)
    print(f"[tfidf+logreg / {a.task}] train={len(Xtr)} test={len(Xte)}", flush=True)

    clf = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=300_000),
        LogisticRegression(max_iter=2000, C=4.0),
    )
    clf.fit(Xtr, ytr)
    P = clf.predict_proba(Xte)
    y = np.array(yte)

    classes = list(clf.classes_)
    order = [classes.index(i) for i in range(len(labels))]
    P = P[:, order]

    pred = P.argmax(1)
    conf = P.max(1)
    correct = (pred == y).astype(float)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0

    f1s = []
    for c in range(len(labels)):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum()); fn = int(((pred != c) & (y == c)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))

    summary = {"model": "tfidf+logreg", "task": a.task, "n": len(y),
               "accuracy": round(float((pred == y).mean()), 4), "macro_f1": round(float(np.mean(f1s)), 4),
               "per_class_f1": {labels[i]: round(float(f1s[i]), 4) for i in range(len(labels))},
               "f1_positive": round(float(f1s[labels.index("true")]), 4) if "true" in labels else None,
               "brier": round(float(((P - onehot) ** 2).sum(1).mean()), 4),
               "nll": round(float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, None)).mean()), 4),
               "ece": round(ece(conf, correct), 4), "mean_confidence": round(float(conf.mean()), 4)}
    for t in (0.9, 0.8):
        m = conf >= t
        summary[f"coverage@{t}"] = round(float(m.mean()), 4)
        summary[f"acc@{t}"] = round(float(correct[m].mean()), 4) if m.sum() else None
    per = {}
    for j in range(len(y)):
        k = labels[y[j]]
        per.setdefault(k, [0, 0]); per[k][1] += 1; per[k][0] += int(pred[j] == y[j])
    summary["per_class_acc"] = {k: round(v[0] / v[1], 4) for k, v in sorted(per.items())}

    out = Path(a.out) if a.out else HERE / "runs" / f"tfidf-logreg__{a.task}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
