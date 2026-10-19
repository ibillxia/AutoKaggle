"""Offline check: does averaging the probabilities of several runs beat the best single run?

Every input is a per-row probability dump from `fit_temperature.py --rows-out` (all dumps must come from the
same labelled rows, in the same order -- prep_llm keeps the holdout order stable across truncation settings, so
dumps made with different `--prompt-chars` still line up).

Each member is calibrated first (its own temperature, optional per-class bias sidecar), then the calibrated
probabilities are averaged with the given weights, then the mixture gets its *own* temperature fit -- that last
step is what makes the comparison honest: every number here is "best temperature for this predictor, chosen on
this holdout", so single members and mixtures are judged under the same amount of fitting.

    python ensemble_probs.py --probs submissions/a.probs.jsonl:1.5874:submissions/a.bias.json \
                             --probs submissions/b.probs.jsonl:1.1677 --weights 1,1
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fit_temperature import fit_bias, fit_temperature, mean_nll  # noqa: E402


def load_spec(spec):
    """`path[:temperature[:bias_json]]` -> (rows, temperature, bias dict)."""
    parts = spec.split(":")
    path = parts[0]
    temp = float(parts[1]) if len(parts) > 1 and parts[1] else 1.0
    bias = {}
    if len(parts) > 2 and parts[2]:
        bias = json.loads(Path(parts[2]).read_text(encoding="utf-8"))["class_bias"]
    rows = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
    return rows, temp, bias


def calibrate(rows, t, bias):
    out = []
    for r in rows:
        v = {k: max(x, 1e-12) ** (1.0 / t) * (2.718281828459045 ** bias.get(k, 0.0)) for k, x in r["p"].items()}
        z = sum(v.values())
        out.append({"label": r["label"], "p": {k: x / z for k, x in v.items()}})
    return out


def accuracy(rows):
    return sum(max(r["p"], key=r["p"].get) == r["label"] for r in rows) / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", action="append", required=True,
                    help="path[:temperature[:bias_json]] -- repeat once per member")
    ap.add_argument("--weights", default="", help="comma-separated member weights (default: equal)")
    ap.add_argument("--bias", type=int, default=0, help="1 = also fit a per-class bias on the mixture")
    a = ap.parse_args()

    members = [load_spec(s) for s in a.probs]
    n = len(members[0][0])
    for path_spec, (rows, _, _) in zip(a.probs, members):
        if len(rows) != n:
            raise SystemExit(f"{path_spec.split(':')[0]} has {len(rows)} rows, expected {n}")
        if [r["label"] for r in rows] != [r["label"] for r in members[0][0]]:
            raise SystemExit(f"{path_spec.split(':')[0]} labels are not aligned with the first dump")
    labels = [r["label"] for r in members[0][0]]

    wts = [float(w) for w in a.weights.split(",")] if a.weights else [1.0] * len(members)
    if len(wts) != len(members):
        raise SystemExit("--weights must have one value per --probs")
    tot = sum(wts)

    print(f"{'member':44s} {'T':>7s} {'NLL(raw)':>9s} {'NLL(T)':>8s} {'acc':>7s}")
    cal = []
    for spec, wt in zip(a.probs, wts):
        rows, t, bias = load_spec(spec)
        c = calibrate(rows, t, bias)
        cal.append((c, wt))
        name = spec.split(":")[0]
        t_fit, nll_fit = fit_temperature(c)
        print(f"{name[:44]:44s} {t:7.4f} {mean_nll(c, 1.0):9.5f} {nll_fit:8.5f} {accuracy(c):7.4f}")

    mix = []
    for i in range(n):
        acc = {}
        for c, wt in cal:
            for k, v in c[i]["p"].items():
                acc[k] = acc.get(k, 0.0) + wt * v
        z = sum(acc.values())
        mix.append({"label": labels[i], "p": {k: v / z for k, v in acc.items()}})

    t_mix, nll_mix = fit_temperature(mix)
    line = (f"{'MIXTURE (%d members)' % len(members):44s} {t_mix:7.4f} {mean_nll(mix, 1.0):9.5f} "
            f"{nll_mix:8.5f} {accuracy(mix):7.4f}")
    print(line)
    if a.bias:
        t2, bias, nll_b = fit_bias(mix, t_mix)
        print(f"{'MIXTURE + per-class bias':44s} {t2:7.4f} {'':9s} {nll_b:8.5f} "
              f"{accuracy(calibrate(mix, t2, bias)):7.4f}   biases "
              f"{ {k: round(v, 4) for k, v in bias.items()} }")

    best_single = min(fit_temperature(c)[1] for c, _ in cal)
    print(f"\ngain of mixture over best single member: {best_single - nll_mix:+.5f} "
          f"({(best_single - nll_mix) / best_single * 100:+.2f}% rel)")
    print("(NLL(raw) = the member's own temperature applied, no refit -- the number comparable to a submission)")


if __name__ == "__main__":
    main()
