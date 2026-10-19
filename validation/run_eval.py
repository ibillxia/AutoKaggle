"""Score a kev checkpoint on the prepared Kaggle-classification requests.

In-process (no HTTP server): Checkpoint.load -> LocalPredictor -> evaluate_records, which is the
same path the project's own benchmarks report from. Then the rows are summarised with the metrics
that matter for a classifier: accuracy, macro-F1, Brier, ECE and latency.

    python run_eval.py --run ../models/kev-0.8b-local --task imdb
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_wincompat"))                 # fcntl shim (Windows)
sys.path.insert(0, "E:/Bill/AI/AutoKaggle/third_party/kev")

import numpy as np
import torch

from kev.benchmark import evaluate_records
from kev.checkpoint import LoadOptions
from kev.data import load_records
from kev.predictors import LocalPredictor
from kev.suite import SERVING_CONTEXT


def ece(conf, correct, bins=15):
    """Expected calibration error, equal-width bins (the definition kev.metrics uses)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total, out = len(conf), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            out += m.sum() / total * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def summarize(rows):
    """Rows carry p (probability per option, in `keys` order) and the integer label."""
    n = len(rows)
    keys = [r["keys"] for r in rows]
    P = np.array([r["p"] for r in rows])
    y = np.array([r["label"] for r in rows])
    pred = P.argmax(1)
    acc = float((pred == y).mean())

    # macro F1 over the option set of each question (all questions share one option set here)
    ks = sorted({k for kk in keys for k in kk})
    f1s = []
    for c in ks:
        idx = [j for j, kk in enumerate(keys) if c in kk]
        if not idx:
            continue
        ci = idx[0]
        tp = sum(1 for j in idx if pred[j] == ci and y[j] == ci)
        fp = sum(1 for j in idx if pred[j] == ci and y[j] != ci)
        fn = sum(1 for j in idx if pred[j] != ci and y[j] == ci)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    macro_f1 = float(np.mean(f1s))

    conf = P.max(1)
    correct = (pred == y).astype(float)
    onehot = np.zeros_like(P)
    for i, (row, kk) in enumerate(zip(rows, keys)):
        onehot[i, kk.index(row["keys"][row["label"]])] = 1.0
    brier = float(((P - onehot) ** 2).sum(1).mean())
    nll = float(-np.log(np.clip(P[np.arange(n), y], 1e-9, None)).mean())

    # selective prediction: accuracy on the questions the model was confident about
    out = {"n": n, "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
           "brier": round(brier, 4), "nll": round(nll, 4), "ece": round(ece(conf, correct), 4),
           "mean_confidence": round(float(conf.mean()), 4)}
    for t in (0.9, 0.8):
        m = conf >= t
        out[f"coverage@{t}"] = round(float(m.mean()), 4)
        out[f"acc@{t}"] = round(float(correct[m].mean()), 4) if m.sum() else None
    per = {}
    for j, kk in enumerate(keys):
        name = kk[y[j]]
        per.setdefault(name, [0, 0])
        per[name][1] += 1
        per[name][0] += int(pred[j] == y[j])
    out["per_class_acc"] = {k: round(v[0] / v[1], 4) for k, v in sorted(per.items())}

    # per-class F1, and the positive-class F1 that nlp-getting-started actually scores on
    pf1 = {}
    for c in sorted({k for kk in keys for k in kk}):
        idx = [j for j, kk in enumerate(keys) if c in kk]
        ci = keys[idx[0]].index(c)
        tp = sum(1 for j in idx if pred[j] == ci and y[j] == ci)
        fp = sum(1 for j in idx if pred[j] == ci and y[j] != ci)
        fn = sum(1 for j in idx if pred[j] != ci and y[j] == ci)
        pf1[c] = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    out["per_class_f1"] = {k: round(v, 4) for k, v in pf1.items()}
    if "true" in pf1:
        out["f1_positive"] = round(pf1["true"], 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", required=True, choices=["imdb", "ag_news", "sms_spam", "disaster", "nli", "xnli", "watson", "watsondev"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    records = load_records(str(HERE / "data" / f"{a.task}_test.jsonl"))
    if a.limit:
        records = records[: a.limit]

    opts = LoadOptions(dtype=torch.bfloat16 if a.dtype == "bf16" else None, backend="torch")
    t0 = time.time()
    predictor = LocalPredictor(a.run, a.device, opts, context=SERVING_CONTEXT)
    load_s = time.time() - t0
    ck = predictor.checkpoint
    tag = f"{Path(a.run).name}"
    print(f"[{tag} / {a.task}] loaded in {load_s:.1f}s on {a.device} (dtype={a.dtype})", flush=True)
    print(f"  base={ck.meta.base}  lora={ck.meta.lora}  T={ck.meta.temperature:.4f}", flush=True)
    if a.device == "cuda":
        print(f"  peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB", flush=True)

    outdir = Path(a.out) if a.out else HERE / "runs" / f"{tag}__{a.task}"
    if outdir.exists():
        import shutil
        shutil.rmtree(outdir)
    t1 = time.time()
    report, rows = evaluate_records(records, predictor, outdir, skip_overlong=True)
    wall = time.time() - t1

    summary = {"model": tag, "run": str(a.run), "task": a.task, "device": a.device, "dtype": a.dtype,
               "params_base": ck.meta.base, "load_s": round(load_s, 1),
               "eval_wall_s": round(wall, 1), "sec_per_row": round(wall / max(len(records), 1), 3),
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if a.device == "cuda" else None,
               **summarize(rows)}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
