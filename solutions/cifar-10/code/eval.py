"""Evaluate trained models and (optionally) write a submission file.

Validation protocol for the 2014 Kaggle cifar-10 competition
-----------------------------------------------------------
Facts established locally (see runs/overlap.log):
  * competition train/ (50k PNG) is byte-identical to the standard CIFAR-10
    train set; labels agree 100%.
  * the 300k competition test images are *not* the standard 10k CIFAR-10 test
    images and are unlabeled, so they can only be scored on Kaggle.
  * the competition is locked since 2014-10-18, so nothing can be uploaded.

Consequently the honest local estimate of leaderboard accuracy is a
cross-validated accuracy on the 50k train set (same distribution the
competition evaluated on), plus the standard 10k test accuracy as an
external sanity check.  `--oof` scores a 5-fold OOF run; `--std` scores the
standard 10k test set; `--predict` writes the submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import LABELS, WideResNet, build_model, evaluate, predict_probs, stratified_folds

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
SUBS = ROOT / "artifacts" / "submissions"


def load_model(ckpt: Path, cfg: dict):
    name = cfg.get("model", "wrn")
    m = build_model(name, cfg.get("depth", 28), cfg.get("widen", 10), cfg.get("drop", 0.0))
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.cuda().to(memory_format=torch.channels_last).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir under runs/, e.g. wrn28-10-e60")
    ap.add_argument("--oof", action="store_true")
    ap.add_argument("--std", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = RUNS / args.run
    cfg = json.loads((run / "config.json").read_text())
    y = np.load(DATA / "npy" / "train_y.npy")

    if args.oof:
        fold = stratified_folds(y, args.folds, cfg.get("seed", args.seed))
        oof = np.zeros((len(y), 10), dtype=np.float32)
        for f in range(args.folds):
            ck = run / f"model_f{f}.pt"
            if not ck.exists():
                print(f"missing {ck}")
                return
            m = load_model(ck, cfg)
            va = np.flatnonzero(fold == f)
            oof[va] = predict_probs(m, np.load(DATA / "npy" / "train_x.npy")[va])
            print(f"fold {f}: {(oof[va].argmax(1)==y[va]).mean():.4f}", flush=True)
            del m
            torch.cuda.empty_cache()
        acc = (oof.argmax(1) == y).mean()
        np.save(run / "oof.npy", oof)
        print(f"OOF accuracy: {acc:.4f}")
        (run / "eval.json").write_text(json.dumps({"oof_acc": float(acc)}))

    if args.std:
        x = np.load(DATA / "npy" / "std_test_x.npy")
        ty = np.load(DATA / "npy" / "std_test_y.npy")
        cks = sorted(run.glob("model_f*.pt")) or [run / "model.pt"]
        accs = {}
        for ck in cks:
            m = load_model(ck, cfg)
            accs[ck.name] = float(evaluate(m, x, ty, tta=True))
            print(f"std test {ck.name} (flip TTA): {accs[ck.name]:.4f}", flush=True)
            del m
            torch.cuda.empty_cache()
        if accs:
            mean = float(np.mean(list(accs.values())))
            print(f"std test mean: {mean:.4f}")
            (run / "std_eval.json").write_text(json.dumps(accs | {"mean": mean}))

    if args.predict:
        xt = np.load(DATA / "npy" / "test_x.npy", mmap_mode="r")
        ids = np.load(DATA / "npy" / "test_ids.npy")
        cks = sorted(run.glob("model_f*.pt"))
        if not cks:
            cks = [run / "model.pt"]
        probs = np.zeros((len(xt), 10), dtype=np.float32)
        for ck in cks:
            m = load_model(ck, cfg)
            probs += predict_probs(m, xt).astype(np.float32)
            print(f"predicted with {ck.name}", flush=True)
            del m
            torch.cuda.empty_cache()
        probs /= len(cks)
        np.save(run / "test_probs.npy", probs.astype(np.float16))
        pred = probs.argmax(1)
        SUBS.mkdir(parents=True, exist_ok=True)
        sub = pd.DataFrame({"id": ids, "label": [LABELS[i] for i in pred]})
        sub.to_csv(SUBS / f"{args.run}.csv", index=False)
        print("class distribution:", np.bincount(pred, minlength=10))
        print(f"wrote {SUBS / (args.run + '.csv')} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
