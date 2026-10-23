"""Convert the Kaggle CIFAR-10 PNG dumps into packed uint8 npy arrays.

train/ has 50000 PNGs named <id>.png with labels in trainLabels.csv.
test/  has 300000 PNGs named <id>.png; submission is id,label for every id.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

LABELS = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]


def load_images(paths, desc):
    out = np.empty((len(paths), 32, 32, 3), dtype=np.uint8)
    t0 = time.time()
    for i, p in enumerate(paths):
        out[i] = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        if (i + 1) % 10000 == 0:
            el = time.time() - t0
            print(f"  {desc} {i+1}/{len(paths)}  {el:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = DATA / "npy"
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.txt").write_text("\n".join(LABELS), encoding="utf-8")

    # ---- train ----
    if args.force or not (out / "train_y.npy").exists():
        lab = pd.read_csv(DATA / "raw" / "trainLabels.csv")
        lab["y"] = lab["label"].map({c: i for i, c in enumerate(LABELS)})
        assert lab["y"].notna().all(), "unknown label"
        lab = lab.sort_values("id").reset_index(drop=True)
        paths = [DATA / "train" / f"{i}.png" for i in lab["id"]]
        print(f"train: {len(paths)} images")
        x = load_images(paths, "train")
        np.save(out / "train_x.npy", x)
        np.save(out / "train_y.npy", lab["y"].to_numpy(np.int64))
        np.save(out / "train_ids.npy", lab["id"].to_numpy(np.int64))
        print("saved train", x.shape)

    # ---- test ----
    if args.force or not (out / "test_x.npy").exists():
        ids = np.array(sorted(int(p.stem) for p in (DATA / "test").glob("*.png")),
                       dtype=np.int64)
        paths = [DATA / "test" / f"{i}.png" for i in ids]
        print(f"test: {len(paths)} images")
        x = load_images(paths, "test")
        np.save(out / "test_x.npy", x)
        np.save(out / "test_ids.npy", ids)
        print("saved test", x.shape)

    x = np.load(out / "train_x.npy", mmap_mode="r")
    y = np.load(out / "train_y.npy")
    print("train", x.shape, np.bincount(y))
    print("test", np.load(out / "test_x.npy", mmap_mode="r").shape)


if __name__ == "__main__":
    main()
