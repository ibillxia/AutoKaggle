"""Local evaluation of a kev checkpoint over any labelled JSONL (accuracy / macro-F1 / log loss).

Written for the two code competitions, whose official test splits are hidden: the number has to come from
a held-out slice of train.csv, so it cannot go through run_eval.py's fixed task table.

    python eval_jsonl.py --run runs/kev-0.8b-ft-watson \
        --data data/comp/contradictory-my-dear-watson/kev_holdout.jsonl --out results/watson.json

Log loss is the metric for llm-classification-finetuning; accuracy for contradictory-my-dear-watson.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "_wincompat"))
sys.path.insert(0, str(ROOT / "third_party" / "kev"))

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--temperature", type=float, default=None, help="override the checkpoint's fitted temperature")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    from kev.suite import SERVING_CONTEXT

    rows = [json.loads(l) for l in Path(a.data).open(encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    opts = LoadOptions(dtype=torch.bfloat16 if a.dtype == "bf16" else None, backend="torch",
                       temperature=a.temperature)
    t0 = time.time()
    predictor = LocalPredictor(a.run, a.device, opts, context=SERVING_CONTEXT)
    print(f"loaded {a.run} in {time.time()-t0:.1f}s (T={predictor.checkpoint.meta.temperature:.4f})", flush=True)

    nll, hits, n = 0.0, 0, 0
    gold_c, pred_c, pair_c = Counter(), Counter(), Counter()
    t1 = time.time()
    for i, rec in enumerate(rows):
        qid, q = next(iter(rec["questions"].items()))
        gold = q["label"]
        probs = predictor(rec)["probabilities"][qid]
        p = max(probs.get(gold, 1e-12), 1e-12)
        nll += -math.log(p)
        pred = max(probs, key=probs.get)
        hits += pred == gold
        n += 1
        gold_c[gold] += 1
        pred_c[pred] += 1
        pair_c[(gold, pred)] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)} acc={hits/n:.4f} logloss={nll/n:.4f} {(time.time()-t1)/(i+1):.3f}s/rec", flush=True)

    acc = hits / n
    labels = sorted(gold_c)
    macro = sum(2 * pair_c[(l, l)] / (gold_c[l] + pred_c[l]) if gold_c[l] + pred_c[l] else 0.0
                for l in labels) / len(labels)
    summary = {"run": a.run, "data": a.data, "n": n, "accuracy": acc, "macro_f1": macro,
               "log_loss": nll / n, "sec_per_row": (time.time() - t1) / n,
               "temperature": predictor.checkpoint.meta.temperature,
               "gold_distribution": dict(gold_c), "pred_distribution": dict(pred_c)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if a.device == "cuda":
        print(f"peak VRAM {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
