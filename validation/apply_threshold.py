"""Re-threshold an existing kev submission by its saved per-row probabilities.

`submission.py predict` writes a `<name>.probs.jsonl` next to the csv. The disaster-tweets metric
is F1 on 0/1 targets, so moving the cut point on `p_true` changes the score without re-running the
model. This re-reads that file and re-emits the submission.

    python apply_threshold.py --probs submissions/kev-0.8b-ft-full.probs.jsonl \
        --threshold 0.45 --comp nlp-getting-started --out submissions/kev-0.8b-ft-full-t45.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SUBMISSION_COL = {
    "nlp-getting-started": ("id", "target"),
    "contradictory-my-dear-watson": ("id", "prediction"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--comp", required=True, choices=sorted(SUBMISSION_COL))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    id_col, val_col = SUBMISSION_COL[a.comp]
    rows, missing = [], 0
    with open(a.probs, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            p = rec.get("p") or {}
            if "true" in p:
                value = 1 if p["true"] >= a.threshold else 0
            else:                                  # choice comp: argmax over label keys
                value = max(p, key=p.get) if p else rec.get("pred")
                missing += 1
            rows.append((rec["id"], value))

    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([id_col, val_col])
        w.writerows(rows)

    pos = sum(1 for _, v in rows if v == 1)
    print(f"wrote {target}: {len(rows)} rows, positives={pos} ({pos / len(rows):.2%}), "
          f"threshold={a.threshold}")
    if missing:
        print(f"note: {missing} rows had no 'true' key and were argmaxed")


if __name__ == "__main__":
    main()
