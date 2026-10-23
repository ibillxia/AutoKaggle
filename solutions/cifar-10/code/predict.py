"""Build the final submission from one or more runs, with flip TTA and
optional multi-model probability averaging.

The 300k test images are predicted in chunks so peak RAM stays small
(test_x.npy is 900MB; it is memory-mapped).

Usage:
  python code/predict.py --runs wrn28-10-s1 wrn28-10-s1-ft40 --tta --name final
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import LABELS, MEAN, STD, build_model, predict_probs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "npy"
RUNS = ROOT / "runs"
SUBS = ROOT / "artifacts" / "submissions"


def variant(name: str, cfg: dict) -> dict:
    """Per-run overrides, e.g. a 40px fine-tune is evaluated at 40px."""
    return {"model": cfg.get("model", "wrn"),
            "depth": cfg.get("depth", 28),
            "widen": cfg.get("widen", 10),
            "drop": cfg.get("drop", 0.0),
            "res": cfg.get("test_res", 32)}


def load(path: Path, v: dict):
    m = build_model(v["model"], v["depth"], v["widen"], v["drop"])
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.cuda().to(memory_format=torch.channels_last).eval()


@torch.no_grad()
def predict(model, xt, res: int, tta: bool, chunk: int = 1024,
            dihedral_tta: bool = False) -> np.ndarray:
    """Full 300k prediction; returns float16 (N,10) probabilities.

    chunk is deliberately small: at 4096 the flip-TTA forward pass peaks near
    7.5GB on an 8GB card, Windows starts paging to shared memory and
    throughput collapses (the same cliff as batch 768 in training).

    dihedral_tta averages over all 8 orientations (h/v flips and 180-rot),
    matching the augmentations the model was trained with. This matters here
    because ~22% of the scored test copies are v-flipped or 180-rotated.
    """
    out = np.empty((len(xt), 10), dtype=np.float16)
    n = len(xt)
    t0 = time.time()
    for i in range(0, n, chunk):
        xb = torch.as_tensor(np.ascontiguousarray(xt[i:i + chunk]),
                             device="cuda").permute(0, 3, 1, 2).float().div_(255.0)
        if res != 32:
            xb = F.interpolate(xb, size=int(round(res * 1.14)), mode="bilinear",
                               align_corners=False, antialias=True)
            xb = xb[..., :res, :res]
        xb = (xb - MEAN.to("cuda")) / STD.to("cuda")

        views = [xb]
        if dihedral_tta:
            views += [torch.flip(xb, dims=[3]),          # h-flip
                      torch.flip(xb, dims=[2]),          # v-flip
                      torch.flip(xb, dims=[2, 3])]       # rot180
        elif tta:
            views += [torch.flip(xb, dims=[3])]

        logits = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for v in views:
                v = v.contiguous(memory_format=torch.channels_last)
                logits = logits + model(v)
        out[i:i + chunk] = F.softmax(logits.float() / len(views),
                                     dim=1).half().cpu()
        if (i // chunk) % 20 == 0:
            done = i + len(xb)
            print(f"  {done}/{n} {done/(time.time()-t0):.0f} img/s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True,
                    help="checkpoint paths, later averaged with earlier")
    ap.add_argument("--configs", nargs="*", default=[],
                    help="config.json per weight (default: infer from run dir)")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--dihedral-tta", action="store_true",
                    help="average all 8 orientations (for --dihedral models)")
    ap.add_argument("--name", default="final")
    ap.add_argument("--save-probs", action="store_true")
    args = ap.parse_args()

    xt = np.load(DATA / "test_x.npy", mmap_mode="r")
    ids = np.load(DATA / "test_ids.npy")
    cfg_paths = args.configs or [
        str(Path(w).parent / "config.json") for w in args.weights]

    probs = np.zeros((len(xt), 10), dtype=np.float32)
    for w, cp in zip(args.weights, cfg_paths):
        cfg = json.loads(Path(cp).read_text())
        v = variant(Path(w).parent.name, cfg)
        print(f"predicting with {w} at {v['res']}px", flush=True)
        m = load(Path(w), v)
        probs += predict(m, xt, v["res"], bool(args.tta),
                         dihedral_tta=bool(args.dihedral_tta)).astype(np.float32)
        del m
        torch.cuda.empty_cache()
    probs /= len(args.weights)

    SUBS.mkdir(parents=True, exist_ok=True)
    if args.save_probs:
        np.save(SUBS / f"{args.name}_probs.npy", probs.astype(np.float16))
    pred = probs.argmax(1)
    sub = pd.DataFrame({"id": ids, "label": [LABELS[i] for i in pred]})
    path = SUBS / f"{args.name}.csv"
    sub.to_csv(path, index=False)
    print("class distribution:", np.bincount(pred, minlength=10).tolist())
    print("mean max prob:", float(probs.max(1).mean()))
    print(f"wrote {path} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
