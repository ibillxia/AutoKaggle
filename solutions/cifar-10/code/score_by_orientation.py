"""Per-orientation breakdown of a submission's local score.

The 53,735 scored test rows are copies of canonical CIFAR-10 test images under
four transforms. A model trained with h-flip augmentation only scores ~0.46 on
the v-flip and rot180 copies, which drags the overall copy accuracy well below
its canonical-test accuracy. This prints the breakdown so that gap is visible
rather than hidden inside one number.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "npy"
RUNS = ROOT / "runs"
LABELS = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]
NAMES = {0: "identity", 1: "h-flip", 2: "v-flip", 3: "rot180"}


def main(sub_csv: str):
    tm = {int(k): tuple(v) for k, v in
          json.loads((RUNS / "test_map.json").read_text()).items()}
    truth = np.load(DATA / "std_test_y.npy")
    sub = pd.read_csv(sub_csv).set_index("id")["label"]
    n2i = {c: i for i, c in enumerate(LABELS)}

    per = defaultdict(lambda: [0, 0])
    votes = defaultdict(lambda: np.zeros(10, dtype=np.int64))
    for pos, (canon, ori) in tm.items():
        p = n2i[sub.loc[pos + 1]]          # ids are 1-based
        t = truth[canon]
        a = per[ori]
        a[0] += int(p == t)
        a[1] += 1
        votes[canon][p] += 1

    print(f"{'orientation':<12}{'n':>8}{'acc':>9}")
    tot_c = tot_n = 0
    for o in sorted(per):
        c, n = per[o]
        tot_c += c
        tot_n += n
        print(f"{NAMES[o]:<12}{n:>8d}{c/n:>9.4f}")
    print(f"{'COPY ACC':<12}{tot_n:>8d}{tot_c/tot_n:>9.4f}")

    covered = np.array(sorted(votes))
    v = np.stack([votes[c] for c in covered])
    voted = v.argmax(1)
    print(f"{'VOTED ACC':<12}{len(covered):>8d}{(voted == truth[covered]).mean():>9.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         str(ROOT / "artifacts" / "submissions" / "wrn_di_dtta.csv"))
