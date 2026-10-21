# Evaluate a kev checkpoint on the grouped holdout: balanced accuracy + confusion dump.
import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "validation" / "_wincompat"))
sys.path.insert(0, str(ROOT / "third_party" / "kev"))

import torch  # noqa: E402

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--dump_preds", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    from kev.suite import SERVING_CONTEXT

    rows = [json.loads(l) for l in Path(a.data).open(encoding="utf-8") if l.strip()]
    opts = LoadOptions(dtype=torch.bfloat16 if a.dtype == "bf16" else None, backend="torch")
    t0 = time.time()
    predictor = LocalPredictor(a.run, a.device, opts, context=SERVING_CONTEXT)
    print(f"loaded {a.run} in {time.time()-t0:.1f}s (T={predictor.checkpoint.meta.temperature:.4f})", flush=True)

    hits, n, nll = 0, 0, 0.0
    gold_c, pred_c, pair_c = Counter(), Counter(), Counter()
    preds = []
    t1 = time.time()
    for i, rec in enumerate(rows):
        qid, q = next(iter(rec["questions"].items()))
        gold = q["label"]
        probs = predictor(rec)["probabilities"][qid]
        p = max(probs.get(gold, 1e-12), 1e-12)
        nll += -math.log(p)
        pred = max(probs, key=probs.get)
        preds.append({"gold": gold, "pred": pred,
                      "p_gold": probs.get(gold, 0.0),
                      "probs": {m: probs.get(m, 0.0) for m in MODELS}})
        hits += pred == gold
        n += 1
        gold_c[gold] += 1
        pred_c[pred] += 1
        pair_c[(gold, pred)] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)} acc={hits/n:.4f} {(time.time()-t1)/(i+1):.3f}s/rec", flush=True)

    recalls = {m: pair_c[(m, m)] / gold_c[m] if gold_c[m] else 0.0 for m in MODELS}
    bal_acc = sum(recalls.values()) / len(MODELS)
    print(f"\nn={n}  accuracy={hits/n:.4f}  balanced_accuracy={bal_acc:.4f}  logloss={nll/n:.4f}")
    print("per-class recall:")
    for m in MODELS:
        bar = "#" * int(recalls[m] * 40)
        print(f"  {m:24s} {recalls[m]:.3f} {bar}")
    # most confused pairs
    conf = sorted(((g, p, c) for (g, p), c in pair_c.items() if g != p), key=lambda x: -x[2])[:10]
    print("top confusions (gold -> pred : count):")
    for g, p, c in conf:
        print(f"  {g} -> {p} : {c}")

    summary = {"run": a.run, "n": n, "accuracy": hits / n, "balanced_accuracy": bal_acc,
               "log_loss": nll / n, "sec_per_row": (time.time() - t1) / n,
               "recalls": recalls}
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if a.dump_preds:
        with open(a.dump_preds, "w", encoding="utf-8") as f:
            for r in preds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
