"""Per-language accuracy for a watson run evaluated on the competition holdout.

The competition is 15 languages and the metric is plain accuracy, so "which language is losing the points"
is the first question any improvement plan has to answer. The competition's own records carry no id, but
their state text reconstructs the exact (premise, hypothesis) pair, which is a key in train.csv -- so the
language column can be joined back without touching the records themselves.

    python lang_breakdown.py --rows validation/runs/watson-ext-holdout/rows.json

`rows.json` is what validation/run_eval.py writes next to summary.json (kev's evaluate_records output).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COMP = ROOT / "data" / "comp" / "contradictory-my-dear-watson"
HOLDOUT = COMP / "kev_holdout.jsonl"

# train.csv spells the language out ("English"), the XNLI configs and kev_dev.meta.json use the two-letter code.
NAME2ABBR = {
    "english": "en", "arabic": "ar", "bulgarian": "bg", "german": "de", "greek": "el",
    "spanish": "es", "french": "fr", "hindi": "hi", "russian": "ru", "swahili": "sw",
    "thai": "th", "turkish": "tr", "urdu": "ur", "vietnamese": "vi", "chinese": "zh",
}


def norm_lang(x):
    """Both spellings collapse to the two-letter code so the competition mix can weight a dev table."""
    if not x:
        return "?"
    x = x.strip()
    return NAME2ABBR.get(x.lower(), x.lower() if len(x) == 2 else x)


def key(p, h):
    return (p.strip(), h.strip())


def load_language_map():
    lang = {}
    for r in csv.DictReader((COMP / "train.csv").open(encoding="utf-8")):
        lang[key(r["premise"], r["hypothesis"])] = norm_lang(r["language"])
    return lang


def holdout_languages():
    """The holdout jsonl order is the order evaluate_records kept (skip_overlong only drops from the tail)."""
    lang = load_language_map()
    out, missing = [], 0
    for line in HOLDOUT.open(encoding="utf-8"):
        rec = json.loads(line)
        body = rec["state"][len("Premise: "):]
        p, h = body.split("\n\nHypothesis: ", 1)
        L = lang.get(key(p, h))
        if L is None:
            missing += 1
            L = "?"
        out.append(L)
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True, help="one or more rows.json files to compare")
    ap.add_argument("--meta", default="",
                    help="per-row {\"lang\":...} list (kev_dev.meta.json) -- use this when the rows come from "
                         "the XNLI/MNLI dev instrument instead of the competition holdout")
    ap.add_argument("--csv", default="", help="also write the table here")
    ap.add_argument("--weight-meta", default="",
                    help="language mix to weight the pooled accuracy by. \"holdout\" = the competition's own "
                         "language distribution (57%% English), which is what a per-language dev set needs: the "
                         "dev set is language-balanced, the competition is not, so pooled accuracy there is not "
                         "comparable to the LB metric.")
    a = ap.parse_args()

    if a.meta:
        langs = [norm_lang(m["lang"]) for m in json.loads(Path(a.meta).read_text(encoding="utf-8"))]
        missing = 0
    else:
        langs, missing = holdout_languages()
    weights = None
    if a.weight_meta:
        if a.weight_meta == "holdout":
            wl, _ = holdout_languages()
            names = [x for x in set(wl) if x != "?"]
            weights = {L: wl.count(L) / max(1, sum(wl.count(x) for x in names)) for L in names}
        else:
            wl = [norm_lang(m["lang"]) for m in json.loads(Path(a.weight_meta).read_text(encoding="utf-8"))]
            names = [x for x in set(wl) if x != "?"]
            weights = {L: wl.count(L) / max(1, sum(wl.count(x) for x in names)) for L in names}
    tables, overall = [], {}
    for path in a.rows:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        # rows can be shorter than the holdout when long records were skipped
        agg = defaultdict(lambda: [0, 0])
        for i, row in enumerate(rows):
            p, y = row["p"], row["label"]
            pred = max(range(len(p)), key=lambda j: p[j])
            L = langs[i] if i < len(langs) else "?"
            agg[L][0] += int(pred == y)
            agg[L][1] += 1
        tables.append((Path(path).parent.name, agg))
        overall[Path(path).parent.name] = sum(v[0] for v in agg.values()) / max(sum(v[1] for v in agg.values()), 1)

    order = sorted({L for _, agg in tables for L in agg}, key=lambda L: -tables[0][1].get(L, [0, 0])[1])
    names = [n for n, _ in tables]
    head = "dev languages" if a.meta else "holdout languages"
    lines = [f"{head} (unmatched: {missing})", "", "lang".ljust(12) + "".join(n.rjust(18) for n in names)]
    lines.append("".ljust(12) + "  acc(n)".rjust(18) * len(names))
    for L in order:
        cells = []
        for _, agg in tables:
            c, n = agg.get(L, [0, 0])
            cells.append(f"{c/n:.3f} ({n})".rjust(18) if n else "—".rjust(18))
        lines.append(L.ljust(12) + "".join(cells))
    lines.append("OVERALL".ljust(12) + "".join(f"{overall[n]:.4f}".rjust(18) for n in names))
    if weights:
        cells = []
        for _, agg in tables:
            acc, wsum = 0.0, 0.0
            for L, w in weights.items():
                c, n = agg.get(L, [0, 0])
                if n:
                    acc += w * c / n
                    wsum += w
            cells.append(f"{acc/wsum:.4f}".rjust(18) if wsum else "—".rjust(18))
        lines.append("COMP-MIX".ljust(12) + "".join(cells))
        lines.append("(COMP-MIX = accuracy reweighted by the competition's own language mix; pooled OVERALL above is"
                     " over a language-balanced set and is not the LB metric)")

    text = "\n".join(lines)
    print(text)
    if a.csv:
        Path(a.csv).write_text(text + "\n", encoding="utf-8")
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
