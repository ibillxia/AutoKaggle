"""Sweep the decision threshold of a binary kev checkpoint on a labelled holdout.

The disaster-tweets competition scores F1 on 0/1 targets, so the score depends on where we cut
the `true` probability. kev already outputs calibrated probabilities, but the F1-optimal cut is
not necessarily 0.5 (F1 favours the minority class when precision is high).

    python tune_threshold.py --run runs/kev-0.8b-ft-disaster \
        --data validation/data/disaster_test.jsonl \
        --out validation/threshold_disaster.json

Writes the sweep table, the best threshold, and a `<out>.probs.jsonl` with per-row `p_true` so the
decision can be re-applied to a real test set without re-running the model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "_wincompat"))
sys.path.insert(0, str(ROOT / "third_party" / "kev"))

import os  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402


def f1_at(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, prec, rec, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True, help="labelled jsonl (one record per line)")
    ap.add_argument("--question", default="disaster")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    from kev.suite import SERVING_CONTEXT

    records = []
    with open(a.data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if a.limit:
        records = records[: a.limit]

    opts = LoadOptions(dtype=torch.bfloat16 if a.dtype == "bf16" else None, backend="torch")
    t0 = time.time()
    predictor = LocalPredictor(a.run, a.device, opts, context=SERVING_CONTEXT)
    print(f"loaded {Path(a.run).name} in {time.time() - t0:.1f}s "
          f"(T={predictor.checkpoint.meta.temperature:.4f})", flush=True)

    y_true, p_true, t1 = [], [], time.time()
    for i, rec in enumerate(records):
        label = rec["questions"][a.question]["label"]
        probs = predictor(rec)["probabilities"][a.question]
        y_true.append(1 if label else 0)
        p_true.append(float(probs.get("true", 0.0)))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(records)}  {(time.time() - t1) / (i + 1):.3f} s/row", flush=True)

    n = len(y_true)
    pos = sum(y_true)
    print(f"\nscored {n} rows in {time.time() - t1:.0f}s; positives={pos} ({pos / n:.1%})", flush=True)

    sweep = []
    for k in range(5, 96, 5):
        thr = k / 100
        pred = [1 if p >= thr else 0 for p in p_true]
        f1, prec, rec, tp, fp, fn = f1_at(y_true, pred)
        acc = sum(1 for t, p in zip(y_true, pred) if t == p) / n
        sweep.append({"threshold": thr, "f1": round(f1, 5), "precision": round(prec, 5),
                      "recall": round(rec, 5), "accuracy": round(acc, 5),
                      "tp": tp, "fp": fp, "fn": fn})

    base = [s for s in sweep if abs(s["threshold"] - 0.5) < 1e-9][0]
    best = max(sweep, key=lambda s: s["f1"])

    print("\nthr    F1      prec    rec     acc")
    for s in sweep:
        mark = " <-- best" if s is best else (" (0.5)" if s is base else "")
        print(f"{s['threshold']:.2f}  {s['f1']:.5f}  {s['precision']:.4f}  "
              f"{s['recall']:.4f}  {s['accuracy']:.4f}{mark}")

    # 最优阈值附近很平，取 F1 在最优 0.2% 以内的所有阈值的中位数更稳（抗单点噪声）
    plateau = [s["threshold"] for s in sweep if s["f1"] >= best["f1"] - 0.002]
    robust = sorted(plateau)[len(plateau) // 2]

    summary = {
        "run": str(a.run),
        "data": str(a.data),
        "n": n, "positives": pos, "positive_rate": round(pos / n, 4),
        "at_0.5": base,
        "best": best,
        "plateau_thresholds": plateau,
        "robust_threshold": robust,
        "gain_over_0.5": round(best["f1"] - base["f1"], 5),
        "sweep": sweep,
    }
    print(f"\nbest thr={best['threshold']:.2f} F1={best['f1']:.5f} "
          f"(0.5 -> {base['f1']:.5f}, gain {best['f1'] - base['f1']:+.5f})")
    print(f"plateau={plateau} -> robust thr={robust:.2f}")

    if a.out:
        target = Path(a.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with target.with_suffix(".probs.jsonl").open("w", encoding="utf-8") as f:
            for t, p in zip(y_true, p_true):
                f.write(json.dumps({"label": t, "p_true": p}) + "\n")
        print(f"wrote {target} and {target.with_suffix('.probs.jsonl')}")


if __name__ == "__main__":
    main()
