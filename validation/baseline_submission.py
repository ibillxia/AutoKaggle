"""TF-IDF + LogReg submission for nlp-getting-started (all 7613 labelled rows, no holdout).

    python baseline_submission.py --out ../submissions/tfidf-logreg.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMP = HERE.parent / "data" / "comp" / "nlp-getting-started"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "submissions" / "tfidf-logreg__nlp-getting-started.csv"))
    a = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    tr = list(csv.DictReader((COMP / "train.csv").open(encoding="utf-8")))
    te = list(csv.DictReader((COMP / "test.csv").open(encoding="utf-8")))
    X = [r["text"] for r in tr]
    y = [int(r["target"]) for r in tr]
    print(f"train={len(X)} test={len(te)} pos_rate={np.mean(y):.4f}")

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
        LogisticRegression(max_iter=2000, C=4.0),
    )
    model.fit(X, y)
    pred = model.predict([r["text"] for r in te])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "target"])
        w.writerows(zip((r["id"] for r in te), (int(p) for p in pred)))
    print(f"pos rate in predictions: {np.mean(pred):.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
