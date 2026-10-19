"""Blend several kev submissions by averaging their per-row probabilities.

Seed/data averaging is the cheapest reliable gain once a single model has plateaued: the members
share an architecture so their errors are correlated, but the uncorrelated part still averages
down. Inputs are the `<name>.probs.jsonl` files written by `submission.py predict`.

    python blend_probs.py --probs submissions/a.probs.jsonl submissions/b.probs.jsonl \
        --weights 1 1 --comp nlp-getting-started --threshold 0.5 --out submissions/blend.csv
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


def load_probs(path: str, key: str) -> dict[str, float]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            p = rec.get("p") or {}
            if key not in p:
                raise SystemExit(f"{path}: row {rec.get('id')} has no '{key}' probability")
            out[rec["id"]] = float(p[key])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--key", default="true", help="probability key to blend (noul: 'true')")
    ap.add_argument("--comp", required=True, choices=sorted(SUBMISSION_COL))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    weights = a.weights or [1.0] * len(a.probs)
    if len(weights) != len(a.probs):
        raise SystemExit("--weights must match --probs in length")
    total = sum(weights)

    maps = [load_probs(p, a.key) for p in a.probs]
    ids = list(maps[0].keys())
    for m, path in zip(maps[1:], a.probs[1:]):
        if set(m) != set(ids):
            raise SystemExit(f"{path}: id set differs from {a.probs[0]}")

    blended = {i: sum(w * m[i] for w, m in zip(weights, maps)) / total for i in ids}

    # how much the members actually disagree
    if len(maps) > 1:
        import statistics
        spread = [max(m[i] for m in maps) - min(m[i] for m in maps) for i in ids]
        print(f"member p_true spread: mean={statistics.mean(spread):.4f} "
              f"p90={sorted(spread)[int(0.9 * len(spread))]:.4f}")

    id_col, val_col = SUBMISSION_COL[a.comp]
    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([id_col, val_col])
        for i in ids:
            w.writerow([i, 1 if blended[i] >= a.threshold else 0])
    with target.with_suffix(".probs.jsonl").open("w", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps({"id": i, "p": {a.key: blended[i]}}) + "\n")

    pos = sum(1 for i in ids if blended[i] >= a.threshold)
    print(f"wrote {target}: {len(ids)} rows, positives={pos} ({pos / len(ids):.2%}), "
          f"threshold={a.threshold}, weights={weights}")


if __name__ == "__main__":
    main()
