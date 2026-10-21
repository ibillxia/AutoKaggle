"""Offline calibration study for the lmsys ensemble.

Consumes `member_probs.npz` produced by kaggle/kernels/lmsys-val (keys
`<member>_pn` for the AB order, `<member>_ps` for the BA order flipped back to
AB space, `<member>_ptta` for their mean) plus the label slice, and reports:

  * per-member log loss in each order (AB / BA / TTA)
  * the public notebook's hand recipe (arithmetic mean, w=0.57, T=1)
  * fitted temperature + blend weight, on a fit/hold-out split
  * vector scaling (per-class gain) as a strictly more general calibration
  * ready-to-paste CALIB dicts for both the TTA and the single-order submission

The single-order dict matters: 25k test rows x 4 forward passes may not fit in the
9h cap, in which case the test kernel falls back to AB-only and needs its own
temperature (averaging two passes flattens the distribution, so the TTA-fitted
temperature is too soft for it).

Everything is a pure function of the npz, so trials take seconds instead of a
50-minute GPU run.  Usage:

    python validation/calib_lmsys.py --npz member_probs.npz --train train.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

LABELS = ["winner_model_a", "winner_model_b", "winner_tie"]


def nll(p, y):
    p = np.clip(p, 1e-15, None)
    p = p / p.sum(1, keepdims=True)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logp(p):
    return np.log(np.clip(p, 1e-12, None))


def fit_T(s, y, grid=np.arange(0.3, 6.005, 0.01)):
    vals = [nll(softmax(s / t), y) for t in grid]
    i = int(np.argmin(vals))
    return float(grid[i]), vals[i]


def fit_vec(k, y):
    """Per-class gain a_c on log p, i.e. p ~ exp(a * log p).  One extra parameter
    per class over temperature, which a tie-heavy metric tends to want."""
    onehot = np.eye(3)[y]

    def loss(a):
        return nll(softmax(k * a), y)

    try:
        from scipy.optimize import minimize
        r = minimize(loss, np.ones(3), method="Nelder-Mead",
                     options=dict(xatol=1e-4, fatol=1e-7, maxiter=2000))
        return r.x, float(r.fun)
    except Exception:  # noqa: BLE001
        a, best = np.ones(3), (loss(np.ones(3)), np.ones(3))
        for _ in range(600):
            a -= 0.05 * ((softmax(k * a) - onehot) * k).mean(0)
            v = loss(a)
            if v < best[0]:
                best = (v, a.copy())
        return best[1], best[0]


def scores(store, n, use_tta=True):
    """Log-prob score of one member.  TTA averages the two response orders in log
    space (= geometric mean); single order keeps the AB pass only."""
    if use_tta:
        return (logp(store[n + "_pn"]) + logp(store[n + "_ps"])) / 2
    return logp(store[n + "_pn"])


def blend(store, names, T, w, kind="arith", A=None, use_tta=True):
    base = {}
    for n in names:
        s = scores(store, n, use_tta)
        base[n] = s * (A[n] if A is not None else 1.0) / T[n]
    if kind == "arith":
        return w * softmax(base[names[0]]) + (1 - w) * softmax(base[names[1]])
    return softmax(w * base[names[0]] + (1 - w) * softmax(base[names[1]]))


def fit_all(store, names, y, use_tta, tag, rows):
    """Fit every calibration variant on `rows` and report."""
    sub = {k: v[rows] for k, v in store.items()}
    yy = y[rows]
    T, A = {}, {}
    for n in names:
        T[n], _ = fit_T(scores(sub, n, use_tta), yy)
        A[n], _ = fit_vec(scores(sub, n, use_tta), yy)
    w1 = {n: 1.0 for n in names}
    out = {}
    for kind in ("arith", "geom"):
        out[kind] = min((nll(blend(sub, names, T, float(w), kind, use_tta=use_tta), yy),
                         float(w)) for w in np.arange(0, 1.001, 0.01))
    vv, wv = min((nll(blend(sub, names, w1, float(w), "arith", A, use_tta=use_tta), yy),
                  float(w)) for w in np.arange(0, 1.001, 0.01))
    print("  [%s] n=%d  T=%s -> arith %.5f (w=%.2f) | geom %.5f (w=%.2f)"
          % (tag, len(rows), {k: round(v, 2) for k, v in T.items()},
             out["arith"][0], out["arith"][1], out["geom"][0], out["geom"][1]))
    print("  [%s] vecscale %.5f (w=%.2f) A=%s"
          % (tag, vv, wv, {k: np.round(v, 2).tolist() for k, v in A.items()}))
    return {"T": T, "w": out["arith"][1], "nll": out["arith"][0],
            "w_geom": out["geom"][1], "nll_geom": out["geom"][0],
            "A": A, "w_vec": wv, "nll_vec": vv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="member_probs.npz")
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--val-n", type=int, default=600)
    ap.add_argument("--names", default="gemma2,llama3")
    a = ap.parse_args()

    z = np.load(a.npz)
    names = [n for n in a.names.split(",") if n + "_pn" in z.files]
    df = pd.read_csv(a.train)
    lab = np.argmax(df[LABELS].values, 1)
    rng = np.random.RandomState(a.seed)
    keep = []
    for c in range(3):
        keep.extend(rng.permutation(np.where(lab == c)[0])[: a.val_n // 3].tolist())
    keep = sorted(keep)
    y = np.argmax(df.iloc[keep][LABELS].values, 1)
    store = {k: z[k] for k in z.files}
    n = len(y)
    m = len(store[names[0] + "_pn"])
    if n != m:
        # keep is sorted, so the npz rows and the label slice are prefix-aligned
        print("WARNING npz has %d rows, label slice %d -- truncating to %d"
              % (m, n, min(m, n)))
        n = min(m, n)
        y = y[:n]
        store = {k: v[:n] for k, v in store.items()}
    print("members %s  rows %d  dist %s" % (names, n, np.bincount(y)))

    print("\n-- members --")
    for nm in names:
        for o in ("pn", "ps", "ptta"):
            k = nm + "_" + o
            if k in store:
                print("  %-7s %-5s nll %.5f  meanmax %.4f"
                      % (nm, o, nll(store[k], y), store[k].max(1).mean()))
        agree = (store[nm + "_pn"].argmax(1) == store[nm + "_ps"].argmax(1)).mean()
        print("  %-7s order agreement %.3f (low = strong position bias)" % (nm, agree))

    print("\n-- hand recipes --")
    print("  uniform                %.5f" % nll(np.full((n, 3), 1 / 3), y))
    print("  class prior            %s"
          % np.round(np.bincount(y, minlength=3) / n, 4))
    for nm in names:
        print("  %s only (TTA)       %.5f" % (nm, nll(store[nm + "_ptta"], y)))
    w1 = {nm: 1.0 for nm in names}
    if len(names) == 2:
        print("  public w=0.57 T=1      %.5f" % nll(blend(store, names, w1, 0.57), y))
        for w in (0.4, 0.5, 0.6, 0.7):
            print("  arith w=%.2f T=1        %.5f" % (w, nll(blend(store, names, w1, w), y)))
    if len(names) < 2:
        fit_all(store, names + names, y, True, "x", np.arange(n))
        return

    # ---- honest split: fit on half, report on the other ----
    perm = np.random.RandomState(7).permutation(n)
    fi, ei = perm[: n // 2], perm[n // 2:]
    print("\n== hold-out check ==")
    for tta in (True, False):
        c = fit_all(store, names, y, tta, "fit %s" % ("TTA" if tta else "AB"), fi)
        ev = {k: v[ei] for k, v in store.items()}
        print("  [eval %s] arith %.5f | geom %.5f | vec %.5f"
              % ("TTA" if tta else "AB",
                 nll(blend(ev, names, c["T"], c["w"], "arith", use_tta=tta), y[ei]),
                 nll(blend(ev, names, c["T"], c["w_geom"], "geom", use_tta=tta), y[ei]),
                 nll(blend(ev, names, {k: 1.0 for k in names}, c["w_vec"], "arith",
                           c["A"], use_tta=tta), y[ei])))

    # ---- final fit on everything ----
    print("\n== full fit (all %d rows) ==" % n)
    res = {}
    for tta in (True, False):
        res[tta] = fit_all(store, names, y, tta, "full %s" % ("TTA" if tta else "AB"),
                           np.arange(n))
    out = {
        "CALIB_TTA": {"T": {k: round(v, 3) for k, v in res[True]["T"].items()},
                      "kind": "arith", "w": res[True]["w"]},
        "CALIB_SINGLE": {"T": {k: round(v, 3) for k, v in res[False]["T"].items()},
                         "kind": "arith", "w": res[False]["w"]},
    }
    print()
    for k, v in out.items():
        print("%s=%s" % (k, json.dumps(v)))
    json.dump(out, open("calib_lmsys.json", "w"), indent=1)
    print("\nwrote calib_lmsys.json")


if __name__ == "__main__":
    main()
