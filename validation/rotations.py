"""Option-order robustness: score every `choice` question under several cyclic option rotations.

LLM-based classifiers are known to shift their answer when the option order changes. kev ships
RotationAveraged for exactly this. Runs on CPU so it does not fight the GPU job.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_wincompat"))
sys.path.insert(0, "E:/Bill/AI/AutoKaggle/third_party/kev")

import numpy as np
import torch

from kev.checkpoint import LoadOptions
from kev.data import load_records
from kev.predictors import LocalPredictor, RotationAveraged
from kev.suite import SERVING_CONTEXT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="../models/kev-0.8b-local")
    ap.add_argument("--task", default="ag_news")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--rotations", type=int, nargs="+", default=[1, 2, 4])
    a = ap.parse_args()

    records = load_records(str(HERE / "data" / f"{a.task}_test.jsonl"))[: a.limit]
    opts = LoadOptions(dtype=None, backend="torch")
    base = LocalPredictor(a.run, "cpu", opts, context=SERVING_CONTEXT)
    from kev.data import materialize

    print(f"{a.task}: {len(records)} rows, {a.rotations} rotation settings", flush=True)
    out, picks = {}, {}
    for n in a.rotations:
        scorer = RotationAveraged(base, n) if n > 1 else base
        ok, flip = 0, 0
        for i, rec in enumerate(records):
            pred = scorer(rec)
            qid = next(iter(rec["questions"]))
            keys = sorted(pred["probabilities"][qid])
            p = np.array([pred["probabilities"][qid][k] for k in keys])
            y = rec["questions"][qid]["label"]
            top = keys[int(p.argmax())]
            ok += int(top == y)
            if n == a.rotations[0]:
                picks[i] = top
            elif top != picks[i]:
                flip += 1
        out[n] = {"accuracy": round(ok / len(records), 4),
                  "rows_whose_answer_changed_vs_rotations_1": flip}
        print(f"  rotations={n}: accuracy={out[n]['accuracy']:.4f}  "
              f"changed_vs_rot1={flip}/{len(records)}", flush=True)

    (HERE / f"rotations_{a.task}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
