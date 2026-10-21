"""Decide the lmsys blend offline from saved member probabilities.

    python validation/blend_gpipe.py <gemma_npz> [<llama_npz>]

Both files hold a (900, 3) probability matrix over the same validation slice.  The
llama side defaults to the serial llama@4096 run in the val512 kernel output, so a
gemma-only probe is enough to answer "is the second member worth 4 hours of GPU".
"""
import glob
import os
import sys

import numpy as np


def nll(p, y):
    p = np.clip(p, 1e-15, None)
    p = p / p.sum(1, keepdims=True)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def acc(p, y):
    return float((p.argmax(1) == y).mean())


def find(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def main():
    gem = sys.argv[1] if len(sys.argv) > 1 else find(
        r"C:\Users\deepg\AppData\Local\Temp\gpipe17\*.npz")
    z = np.load(gem)
    b = z["gemma2_pn"] if "gemma2_pn" in z.files else z[z.files[0]]
    y = z["y"] if "y" in z.files else None

    ref = sys.argv[2] if len(sys.argv) > 2 else find(
        r"C:\Users\deepg\AppData\Local\Temp\val512out\member_probs.npz")
    r = np.load(ref)
    a = r["llama3_pn"]
    if y is None:
        raise SystemExit("no labels in %s" % gem)
    assert len(a) == len(b) == len(y), (len(a), len(b), len(y))

    print("gemma2@4096 : nll %.5f  acc %.3f" % (nll(b, y), acc(b, y)))
    print("llama3@4096 : nll %.5f  acc %.3f" % (nll(a, y), acc(a, y)))
    lo = np.clip(a, 1e-9, None)
    lg = np.clip(b, 1e-9, None)
    print("mean KL(gemma||llama) %.4f | argmax agree %.3f"
          % (float((lg * np.log(lg / lo)).sum(1).mean()),
             float((a.argmax(1) == b.argmax(1)).mean())))

    print("\n  w_gemma   nll      acc")
    best = (9.9, 0.0)
    for w in [i / 100.0 for i in range(0, 61, 5)]:
        p = w * b + (1 - w) * a
        v = nll(p, y)
        best = min(best, (v, w))
        print("   %.2f    %.5f  %.3f%s" % (w, v, acc(p, y), "  <==" if w == 0 else ""))

    # 5-fold over the weight grid: a single 900-row number overfits the weight
    idx = np.arange(len(y))
    folds = np.random.RandomState(7).permutation(idx) % 5
    rows = []
    for w in [i / 100.0 for i in range(0, 51, 2)]:
        tot = sum(nll(w * b[te] + (1 - w) * a[te], y[te]) * len(te)
                  for te in (idx[folds == f] for f in range(5)))
        rows.append((tot / len(idx), w))
    rows.sort()
    base = [v for v, w in rows if w == 0.0][0]
    print("\ncv best w=%.2f (%.5f) vs w=0 (%.5f) -> delta %+.5f"
          % (rows[0][1], rows[0][0], base, rows[0][0] - base))
    print("in-sample best w=%.2f (%.5f)" % (best[1], best[0]))


if __name__ == "__main__":
    main()
