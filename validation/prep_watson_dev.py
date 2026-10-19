"""Build a *low-noise* multilingual NLI dev set for choosing between watson candidates.

`kev_holdout.jsonl` is only 1,213 rows (1 sigma on accuracy ~= 1.1 pt), which is wider than most of the
differences we are trying to resolve, and its low-resource languages have 30-45 rows each -- useless for the
per-language question that drives the external-data work. Meanwhile the competition's 5,195-row public test can
only be used once per submission.

The competition is drawn from XNLI + MNLI, whose *train* splits we already train on, so their **dev** splits are
clean, large and language-labelled: XNLI validation covers all 15 languages (~2,490 rows each) and MNLI's
validation_matched is the natural English source. With ~1,000 rows per language the per-language sigma drops to
~1.2 pt and the pooled sigma to ~0.35 pt.

Rows whose (premise, hypothesis) already appear in the competition's own train.csv or in our holdout are dropped,
so no evaluated row was trained on.

    python prep_watson_dev.py --out data/comp/contradictory-my-dear-watson/kev_dev.jsonl

Writes the JSONL plus `kev_dev.meta.json` (one {"lang", "src"} per row, in row order) so `lang_breakdown.py`
can group without re-deriving the language from the text.
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # everything below is already in the local cache
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COMP = ROOT / "data" / "comp" / "contradictory-my-dear-watson"

CRITERIA = ["entailment", "neutral", "contradiction"]
INSTRUCTIONS = "What is the logical relationship between the premise and the hypothesis?"
XNLI_LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]


def rec(premise, hypothesis, label, src, lang):
    return {
        "state": "Premise: %s\n\nHypothesis: %s" % (premise.strip(), hypothesis.strip()),
        "questions": {
            "relation": {"type": "choice", "instructions": INSTRUCTIONS,
                         "criteria": {k: k for k in CRITERIA},
                         "label": CRITERIA[int(label)], "src": "%s/%s" % (src, lang)},
        },
    }


def balanced(rows, n, rng):
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[int(r["label"])].append(i)
    per = max(1, n // 3)
    idx = []
    for lab in sorted(by):
        pool = by[lab]
        rng.shuffle(pool)
        idx += pool[:per]
    rng.shuffle(idx)
    return [rows[i] for i in idx]


def key(p, h):
    return (p.strip(), h.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=1000, help="XNLI dev rows per non-English language")
    ap.add_argument("--mnli", type=int, default=3000, help="natural English MNLI dev_matched rows")
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--out", default=str(COMP / "kev_dev.jsonl"))
    a = ap.parse_args()

    from datasets import load_dataset
    rng = random.Random(a.seed)

    # anything the model has already seen, plus anything the competition itself publishes as a test row.
    # Both matter: `train.csv` and `holdout.jsonl` are training data, and `test.csv` carries the full text of
    # every public test row -- an instrument that contains them measures the public LB, not generalisation.
    seen = set()
    for name in ("kev_train_split.jsonl", "kev_holdout.jsonl"):
        for line in (COMP / name).open(encoding="utf-8"):
            r = json.loads(line)
            body = r["state"][len("Premise: "):]
            p, h = body.split("\n\nHypothesis: ", 1)
            seen.add(key(p, h))
    import csv
    for r in csv.DictReader((COMP / "test.csv").open(encoding="utf-8")):
        seen.add(key(r["premise"], r["hypothesis"]))
    print("rows excluded (competition train/holdout/test): %d" % len(seen))

    out, meta = [], []
    dropped = 0

    for lang in XNLI_LANGS:
        if lang == "en":
            continue
        try:
            ds = load_dataset("facebook/xnli", lang, split="validation")
        except Exception as exc:  # noqa: BLE001
            print("  xnli %-2s: SKIPPED (%s: %s)" % (lang, type(exc).__name__, str(exc)[:100]))
            continue
        rows = [{"premise": ds[i]["premise"], "hypothesis": ds[i]["hypothesis"], "label": ds[i]["label"]}
                for i in range(len(ds))]
        keep = []
        for r in balanced(rows, a.per_lang, rng):
            if key(r["premise"], r["hypothesis"]) in seen:
                dropped += 1
                continue
            keep.append(r)
        out += [rec(r["premise"], r["hypothesis"], r["label"], "xnli", lang) for r in keep]
        meta += [{"lang": lang, "src": "xnli"} for _ in keep]
        print("  xnli %-2s: %d rows" % (lang, len(keep)))

    # English: the competition's own English rows all come from MNLI's *matched* dev split (measured: 2090 of
    # train.csv and 2945 of test.csv are dev_matched rows), so after excluding the competition's own rows almost
    # nothing of dev_matched is left. dev_mismatched is the natural English instrument instead: same corpus, same
    # label set, different genre mix, and it overlaps neither train.csv nor test.csv at all.
    for split in ("validation_mismatched", "validation_matched"):
        try:
            ds = load_dataset("nyu-mll/glue", "mnli", split=split)
        except Exception as exc:  # noqa: BLE001
            print("  mnli %-21s: SKIPPED (%s: %s)" % (split, type(exc).__name__, str(exc)[:100]))
            continue
        rows = [{"premise": ds[i]["premise"], "hypothesis": ds[i]["hypothesis"], "label": ds[i]["label"]}
                for i in range(len(ds))]
        keep = []
        for r in balanced(rows, a.mnli, rng):
            if key(r["premise"], r["hypothesis"]) in seen:
                dropped += 1
                continue
            keep.append(r)
        out += [rec(r["premise"], r["hypothesis"], r["label"], "mnli", "en") for r in keep]
        meta += [{"lang": "en", "src": "mnli-%s" % split} for _ in keep]
        print("  mnli  en (%s): %d rows" % (split, len(keep)))

    assert len(out) == len(meta)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    by_lang = defaultdict(int)
    for m in meta:
        by_lang[m["lang"]] += 1
    print("wrote %d dev rows -> %s  (dropped %d overlapping, %d languages)"
          % (len(out), a.out, dropped, len(by_lang)))


if __name__ == "__main__":
    main()
