"""Fit one temperature on a labelled JSONL and write it into a run's head.pt.

`kev.train` never fits a temperature (unlike the released checkpoints, whose head.pt carries one fitted on held-out
rows), so a delta fine-tune serves raw logits at T=1. That is harmless for an accuracy/F1 metric -- temperature
never changes the argmax -- and decisive for a log-loss metric like llm-classification-finetuning.

The checkpoint's own `scripts/calibrate_checkpoint.py` refuses to fit on rows its infra cannot place, which every
competition holdout is, so this is the small local equivalent: read raw probabilities at T=1, grid-search the
temperature that minimises mean NLL, write it back.

    python fit_temperature.py --run runs/kev-0.8b-ft-llm \\
        --data data/comp/llm-classification-finetuning/kev_holdout.jsonl \\
        --rows-out submissions/llm_holdout.probs.jsonl
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "_wincompat"))
sys.path.insert(0, str(ROOT / "third_party" / "kev"))

import torch  # noqa: E402


def mean_nll(probs_rows, t=1.0, bias=None):
    """-log p_label after p ∝ p^(1/T) * exp(bias). `bias=None` is the plain temperature fit."""
    s = 0.0
    for r in probs_rows:
        if bias:
            v = {k: max(x, 1e-12) ** (1.0 / t) * math.exp(bias[k]) for k, x in r["p"].items()}
        else:
            v = {k: max(x, 1e-12) ** (1.0 / t) for k, x in r["p"].items()}
        z = sum(v.values())
        s += -math.log(max(v[r["label"]] / z, 1e-12))
    return s / len(probs_rows)


def fit_temperature(probs_rows, n_grid=121):
    """The same 121-point log grid 0.25..4 the released fit uses; keeps 1.05^0.5 behaviour simple."""
    best, best_t = None, None
    for k in range(n_grid):
        t = 0.25 * (16.0 ** (k / (n_grid - 1)))
        v = mean_nll(probs_rows, t)
        if best is None or v < best:
            best, best_t = v, t
    return best_t, best


def fit_bias(probs_rows, t0, rounds=4, tol=1e-7, t_hi=8.0):
    """Vector scaling: coordinate descent on T and one additive bias per class.

    Returns (t, bias, nll). The bias is centred (it is only identified up to a constant) so it can be applied
    as a plain `p_c <- p_c * exp(b_c); renormalise` after the served temperature. T is kept inside
    [0.25/T_HI] because with free biases a very large T plus a large bias can mimic a much cheaper solution
    and drifts away from anything the grid fit justified.
    """
    keys = sorted(probs_rows[0]["p"])
    b = {k: 0.0 for k in keys}
    t, cur = t0, mean_nll(probs_rows, t0, b)
    step = 0.5
    while step > 0.005:
        improved = False
        for cand in (t - step, t + step):
            if 0.1 <= cand <= t_hi:
                v = mean_nll(probs_rows, cand, b)
                if v < cur - tol:
                    cur, t, improved = v, cand, True
        for k in keys:
            for d in (step, -step):
                b2 = dict(b)
                b2[k] += d
                v = mean_nll(probs_rows, t, b2)
                if v < cur - tol:
                    cur, b, improved = v, b2, True
        if not improved:
            step /= 2
    mean_b = sum(b.values()) / len(b)
    return t, {k: v - mean_b for k, v in b.items()}, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--rows-out", default="", help="dump the raw (T=1) probabilities per row")
    ap.add_argument("--out", default="", help="write a small json summary here")
    ap.add_argument("--write", type=int, default=1, help="1 = store the fitted temperature in head.pt")
    ap.add_argument("--write-as", default="", help="write head.pt into this copy of the run dir instead (the source stays pristine)")
    ap.add_argument("--bias", type=int, default=0,
                    help="1 = also fit a per-class additive bias after the temperature (Vector scaling / "
                         "multi-class logistic recalibration). Cannot be baked into head.pt -- kev's serving "
                         "path applies the temperature only -- so the biases are printed and written to "
                         "--rows-out's sidecar for the kernel to apply post-hoc.")
    a = ap.parse_args()

    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    from kev.suite import SERVING_CONTEXT

    rows = [json.loads(l) for l in Path(a.data).open(encoding="utf-8") if l.strip()]
    opts = LoadOptions(dtype=torch.bfloat16 if a.dtype == "bf16" else None, backend="torch", temperature=1.0)
    t0 = time.time()
    predictor = LocalPredictor(a.run, a.device, opts, context=SERVING_CONTEXT)
    print(f"loaded {a.run} in {time.time()-t0:.1f}s (served T was {predictor.checkpoint.meta.temperature:.4f})", flush=True)

    probs_rows = []
    for i, rec in enumerate(rows):
        qid, q = next(iter(rec["questions"].items()))
        probs = predictor(rec)["probabilities"][qid]
        probs_rows.append({"label": q["label"], "p": probs})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    raw = mean_nll(probs_rows, 1.0)
    best_t, best = fit_temperature(probs_rows)
    print(f"raw NLL {raw:.5f} -> fitted T {best_t:.4f} gives NLL {best:.5f}  (ln K = {math.log(len(probs_rows[0]['p'])):.5f})")

    # ---- optional second stage: per-class additive bias on top of the fitted temperature ---------------
    # p'_c ∝ p_c^(1/T) * exp(b_c). The temperature can only flatten/sharpen; if the model's class prior is
    # off (arena ties are common), a bias vector removes a systematic term the temperature cannot touch.
    bias = {k: 0.0 for k in probs_rows[0]["p"]}
    best_bias_nll = None
    if a.bias:
        best_t, bias, best_bias_nll = fit_bias(probs_rows, best_t)
        print(f"  + per-class bias: T -> {best_t:.4f}, NLL {best:.5f} -> {best_bias_nll:.5f} "
              f"({(best - best_bias_nll) / best * 100:+.2f}% rel), biases {({k: round(v, 4) for k, v in bias.items()})}")

    if a.rows_out:
        Path(a.rows_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.rows_out, "w", encoding="utf-8") as f:
            for r in probs_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("wrote", a.rows_out)
        if a.bias:
            side = Path(a.rows_out).with_suffix(".bias.json")
            side.write_text(json.dumps({"temperature": best_t, "class_bias": bias,
                                        "n": len(probs_rows), "fitted_nll": best_bias_nll}, indent=2),
                            encoding="utf-8")
            print("wrote", side, "-- the kernel must apply exp(bias) after the temperature")

    if a.write:
        from kev.checkpoint import read_meta, write_meta
        target = a.write_as or a.run
        meta = read_meta(a.run)
        meta.temperature = best_t
        meta.extra["temperature_fit"] = {"rows": a.data, "n": len(probs_rows), "method": "min mean NLL over a 121-point log grid 0.25..4",
                                         "raw_nll": raw, "fitted_nll": best, "temperature": best_t}
        if a.write_as:
            os.makedirs(target, exist_ok=True)
            for f in os.listdir(a.run):
                src, dst = Path(a.run) / f, Path(target) / f
                if src.is_file() and not dst.exists():
                    dst.write_bytes(src.read_bytes())
        write_meta(target, meta)
        print(f"wrote T={best_t:.4f} into {target}/head.pt")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"run": a.run, "data": a.data, "n": len(probs_rows),
                                           "raw_nll": raw, "temperature": best_t, "fitted_nll": best,
                                           "class_bias": bias if a.bias else None,
                                           "fitted_nll_with_bias": best_bias_nll},
                                          indent=2), encoding="utf-8")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
