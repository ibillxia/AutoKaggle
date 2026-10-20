"""Mix per-member test probabilities with per-member temperature calibration.

    python mix_test_probs.py --member submissions/xlmr_big_test_probs.jsonl:1.382:0.5 \
                             --member submissions/xlmr_full_test_probs.jsonl:1.231:0.5 \
                             --out submissions/mix_test.csv

Temperatures come from analysis_watson_dev.py (fit on the dev instrument).  Mixing in
log-space (log p / T) is identical to softmax(logits / T) up to a constant shift, so the
softmax over `log p / T` reproduces the dev-time recipe exactly.  Argmax is invariant to
the mixture-level temperature, so only the per-member T matters here.
"""
import argparse
import csv
import json
import math
from pathlib import Path

CLASSES = ["entailment", "neutral", "contradiction"]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows.append((r["id"], [float(x) for x in r["p"]]))
    return rows


def temper(p, T):
    lg = [math.log(max(x, 1e-12)) / T for x in p]
    m = max(lg)
    e = [math.exp(x - m) for x in lg]
    s = sum(e)
    return [x / s for x in e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True, help="path:T:weight")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    a = ap.parse_args()

    members = []
    for spec in a.member:
        path, T, w = spec.rsplit(":", 2)
        members.append((load(path), float(T), float(w), Path(path).name))
    n = len(members[0][0])
    for rows, _, _, name in members:
        if len(rows) != n:
            raise SystemExit("%s has %d rows, expected %d" % (name, len(rows), n))
        if [r[0] for r in rows] != [r[0] for r in members[0][0]]:
            raise SystemExit("%s id order differs from the first member" % name)
    print("members %d  rows %d  weights %s"
          % (len(members), n, [m[2] for m in members]), flush=True)
    wsum = sum(m[2] for m in members)

    out_rows = []
    counts = {c: 0 for c in CLASSES}
    agree = 0
    for i in range(n):
        acc = [0.0, 0.0, 0.0]
        for rows, T, w, _ in members:
            for k, v in enumerate(temper(rows[i][1], T)):
                acc[k] += (w / wsum) * v
        lab = max(range(3), key=lambda k: acc[k])
        counts[CLASSES[lab]] += 1
        # the competition expects the integer class index, not the label string
        out_rows.append((members[0][0][i][0], lab))
        if lab == max(range(3), key=lambda k: members[0][0][i][1][k]):
            agree += 1

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "prediction"])
        wr.writerows(out_rows)
    print("wrote %s  %s  agrees with member0 on %.3f"
          % (a.out, counts, agree / n), flush=True)

    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            for i, lab in out_rows:
                fh.write(json.dumps({"id": i, "prediction": lab}) + "\n")


if __name__ == "__main__":
    main()
