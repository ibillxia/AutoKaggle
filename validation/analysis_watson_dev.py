"""Offline watson analysis on the kev_dev instrument: per-language COMP-MIX, and does averaging
ext (best English) with pool (best non-English) beat either alone?"""
import json
import sys
from pathlib import Path

ROOT = Path("E:/Bill/AI/AutoKaggle")
sys.path.insert(0, str(ROOT / "validation"))
from fit_temperature import fit_temperature, mean_nll  # noqa: E402

DEV = ROOT / "data/comp/contradictory-my-dear-watson"
DEFAULT_RUNS = ["kev-0.8b-ft-watson-delta", "kev-0.8b-ft-watson-ext", "kev-0.8b-ft-watson-pool"]
RUNS = list(DEFAULT_RUNS)
if "--runs" in sys.argv:
    RUNS = sys.argv[sys.argv.index("--runs") + 1].split(",")
meta = json.loads((DEV / "kev_dev.meta.json").read_text(encoding="utf-8"))
langs = [m["lang"] for m in meta]


def load(run):
    rows = json.loads((ROOT / "validation/runs" / f"{run}-dev" / "rows.json").read_text(encoding="utf-8"))
    out = []
    for r in rows:
        keys = r["keys"]
        out.append({"label": keys[r["label"]], "p": {k: float(v) for k, v in zip(keys, r["p"])}})
    return out


def comp_mix_probs():
    """competition's own language mix, renormalised over the languages the instrument covers"""
    import csv
    from collections import Counter
    c = Counter()
    for r in csv.DictReader((DEV / "test.csv").open(encoding="utf-8")):
        c[r["language"].strip().lower()] += 1
    ab = {"english": "en", "arabic": "ar", "bulgarian": "bg", "german": "de", "greek": "el",
          "spanish": "es", "french": "fr", "hindi": "hi", "russian": "ru", "swahili": "sw",
          "thai": "th", "turkish": "tr", "urdu": "ur", "vietnamese": "vi", "chinese": "zh"}
    w = {ab[k]: v for k, v in c.items()}
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


W = comp_mix_probs()


def stats(rows):
    n = len(rows)
    acc = sum(1 for r in rows if max(r["p"], key=r["p"].get) == r["label"]) / n
    t, _ = fit_temperature(rows)
    return {"acc": acc, "raw_nll": mean_nll(rows), "t": t, "nll": mean_nll(rows, t)}


def per_lang_acc(rows, cmix=False):
    acc = {L: [0, 0] for L in W}
    for r, L in zip(rows, langs):
        acc.setdefault(L, [0, 0])
        acc[L][0] += int(max(r["p"], key=r["p"].get) == r["label"])
        acc[L][1] += 1
    table = {L: (c / n if n else None, n) for L, (c, n) in acc.items()}
    if cmix:
        tot, wsum = 0.0, 0.0
        for L, w in W.items():
            if table.get(L) and table[L][1]:
                tot += w * table[L][0]
                wsum += w
        return table, tot / wsum
    return table, None


def calibrate(rows, t):
    out = []
    for r in rows:
        v = {k: max(x, 1e-12) ** (1.0 / t) for k, x in r["p"].items()}
        z = sum(v.values())
        out.append({"label": r["label"], "p": {k: x / z for k, x in v.items()}})
    return out


def blend(rows_list, weights):
    out = []
    for tup in zip(*rows_list):
        p = {}
        for r, w in zip(tup, weights):
            for k, x in r["p"].items():
                p[k] = p.get(k, 0.0) + w * x
        z = sum(p.values())
        out.append({"label": tup[0]["label"], "p": {k: x / z for k, x in p.items()}})
    return out


def main():
    data = {r: load(r) for r in RUNS}
    print("== single members (instrument: 22154 rows, language-balanced) ==")
    mixed = {}
    for r in RUNS:
        rows = data[r]
        tbl, cm = per_lang_acc(rows, cmix=True)
        s = stats(rows)
        mixed[r] = (calibrate(rows, s["t"]), s["t"])
        print("%-28s acc %.4f  COMP-MIX %.4f  rawNLL %.4f  T %.3f  NLL %.4f"
              % (r.replace("kev-0.8b-ft-watson-", ""), s["acc"], cm, s["raw_nll"], s["t"], s["nll"]))

    print("\n== mixtures (each member calibrated with its own T, mixture then refit) ==")
    import itertools
    for a, b in itertools.combinations(RUNS, 2):
        for wa in (0.5, 0.3, 0.7):
            m = blend([mixed[a][0], mixed[b][0]], [wa, 1 - wa])
            tbl, cm = per_lang_acc(m, cmix=True)
            s = stats(m)
            print("  %-14s + %-14s w=%.1f/%.1f  acc %.4f  COMP-MIX %.4f  NLL %.4f (T %.3f)"
                  % (a.replace("kev-0.8b-ft-watson-", ""), b.replace("kev-0.8b-ft-watson-", ""),
                     wa, 1 - wa, s["acc"], cm, s["nll"], s["t"]))
    m = blend([mixed[r][0] for r in RUNS], [1, 1, 1])
    tbl, cm = per_lang_acc(m, cmix=True)
    s = stats(m)
    print("  ALL3                              acc %.4f  COMP-MIX %.4f  NLL %.4f" % (s["acc"], cm, s["nll"]))

    print("\n== per-language COMP-MIX for the three members ==")
    tables = {r: per_lang_acc(data[r])[0] for r in RUNS}
    print("lang".ljust(5) + "weight".rjust(8) + "".join(r.replace("kev-0.8b-ft-watson-", "").rjust(10) for r in RUNS))
    best_acc, wsum = 0.0, 0.0
    for L in sorted(W, key=lambda L: -W[L]):
        if L not in tables[RUNS[0]]:
            continue
        cells = "".join(("%.3f" % tables[r][L][0]).rjust(10) for r in RUNS)
        b = max(tables[r][L][0] for r in RUNS)
        best_acc += W[L] * b
        wsum += W[L]
        print(L.ljust(5) + ("%.3f" % W[L]).rjust(8) + cells + "   best %.3f" % b)
    print("ORACLE (per-language best of the three, weighted) = %.4f" % (best_acc / wsum))


if __name__ == "__main__":
    main()
