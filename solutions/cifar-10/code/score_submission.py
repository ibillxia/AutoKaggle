"""Local scoring harness for the 2014 Kaggle cifar-10 test set.

What the test set really is (verified locally, see runs/test_map.json):
  300,000 images, of which 53,735 are pixel-exact copies -- identity,
  h-flip, v-flip or 180-rotation -- of 9,957 of the 10,000 canonical CIFAR-10
  test images.  The remaining ~246k are junk that Kaggle ignores when
  scoring (competition data page: "we have added 290,000 junk images in the
  test set. These images are ignored in the scoring").

So we can score a submission *locally* almost exactly as Kaggle does:
the scored positions are the copies of canonical test images, and their true
labels are the canonical labels.

Metric, two variants:
  * `copy_accuracy` -- accuracy over the 53,735 scored rows. This mirrors
    "percentage of correctly categorized items" computed per scored image and
    is the **conservative** estimate.
  * `canonical_accuracy` -- majority vote per distinct canonical image (9,957
    of them), then accuracy. Because each original appears ~5.4 times, voting
    removes independent errors, so this runs *optimistically*; it is an upper
    bound, useful as a stability check rather than the headline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "npy"
RUNS = ROOT / "runs"
LABELS = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]
ORIENT = ["id", "hflip", "vflip", "rot180"]


def load_map():
    m = json.loads((RUNS / "test_map.json").read_text())
    return {int(k): tuple(v) for k, v in m.items()}


def score_submission(sub_csv: Path, label_col: str = "label", verbose: bool = True) -> dict:
    import pandas as pd

    test_map = load_map()
    truth = np.load(DATA / "std_test_y.npy")
    test_ids = np.load(DATA / "npy" / "test_ids.npy") if (DATA / "npy").exists() else np.load(DATA / "test_ids.npy")

    sub = pd.read_csv(sub_csv)
    assert set(sub.columns) >= {"id", label_col}, sub.columns
    assert len(sub) == 300000, f"expected 300000 rows, got {len(sub)}"
    assert sub["id"].is_unique, "duplicate ids"
    name2idx = {c: i for i, c in enumerate(LABELS)}

    pred = sub.set_index("id")[label_col]
    # test_map keys are 0-based row positions into test_x.npy; submission ids
    # are 1..300000, so the id of position p is p + 1. (Getting this wrong
    # scores ~0.10 instead of ~0.96 -- every prediction shifted by one row.)
    sub_ids = [p + 1 for p in test_map]
    missing = set(sub_ids) - set(pred.index)
    assert not missing, f"{len(missing)} scored positions missing from submission"

    # per-copy correctness
    rows = []
    for pos, (canon, orient) in test_map.items():
        rows.append((canon, name2idx[pred.loc[pos + 1]]))
    arr = np.array(rows)
    canon, p = arr[:, 0].astype(int), arr[:, 1].astype(int)

    copy_acc = float((p == truth[canon]).mean())

    # majority vote per canonical image, ties -> lowest copy index
    order = np.argsort([pos for pos in test_map], kind="stable")
    voted = np.empty(len(truth), dtype=np.int64)
    voted[:] = -1
    votes = np.zeros((len(truth), 10), dtype=np.int64)
    np.add.at(votes, (canon, p), 1)
    covered = np.flatnonzero(votes.sum(1) > 0)
    voted[covered] = votes[covered].argmax(1)
    canon_acc = float((voted[covered] == truth[covered]).mean())

    res = {
        "submission": str(sub_csv),
        "copy_accuracy": copy_acc,
        "canonical_accuracy_voted": canon_acc,
        "n_canonical_covered": int(len(covered)),
        "n_copies": int(len(canon)),
        "pred_class_hist": np.bincount(p, minlength=10).tolist(),
    }
    if verbose:
        print(json.dumps(res, indent=1))
        print(f"\n>> estimated Kaggle score (per scored copy, conservative): {copy_acc:.5f}")
        print(f">> upper bound (majority vote over 9957 originals):          {canon_acc:.5f}")
        print(">> 2014 leaderboard: #1 0.95530 | top-10% cutoff 0.89930")
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    a = ap.parse_args()
    score_submission(Path(a.submission))
