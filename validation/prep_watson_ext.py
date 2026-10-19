"""Augment the watson train split with the external corpora the competition is drawn from.

The competition's own `train.csv` is only 12,120 rows, but its rules explicitly allow External Data
(Rule C) and the competition data itself is licensed as *"The Cross-Lingual NLI Corpus and The Multi-Genre
NLI Corpus"* -- i.e. it is a subsample of XNLI + MNLI. So the natural augmentation is to pull the **train**
splits of those two corpora (never their dev/test splits, which is where the competition's own hidden test
almost certainly lives) and mix them with the competition train split.

Language mix follows the competition's own distribution (English ≈ 57% of test rows), so English gets the
natural MNLI rows plus extra XNLI-en, and the other 14 languages get a uniform per-language quota.

    python prep_watson_ext.py --out data/comp/contradictory-my-dear-watson/kev_train_ext.jsonl

Writes a JSONL of kev `choice` records; the competition holdout (kev_holdout.jsonl) is never touched.
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
    """`n` rows with the three labels as evenly as the source allows (the competition's own split is balanced)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=800, help="XNLI rows per non-English language")
    ap.add_argument("--xnli-en", type=int, default=0,
                    help="XNLI-en rows. 0 by default on purpose: XNLI's train split IS MNLI, so XNLI-en rows "
                         "are near-duplicates of --mnli rows (verified: same premises, same order). English "
                         "comes from MNLI instead.")
    ap.add_argument("--mnli", type=int, default=6000, help="natural English MNLI rows")
    ap.add_argument("--comp", type=int, default=0, help="competition train-split rows (0 = all of kev_train_split)")
    ap.add_argument("--comp-repeat", type=int, default=1,
                    help="write the competition rows this many times, to upweight the in-domain data. The "
                         "external corpora here are the *train* splits of XNLI/MNLI, while the competition was "
                         "carved out of their dev-side pools (measured: 56.7%% of the public test is in MNLI "
                         "dev_matched), so the competition's own rows are the closest thing to the test "
                         "distribution and are worth more per step than a corpus row.")
    ap.add_argument("--comp-file", default="kev_train_split.jsonl",
                    help="which competition file to mix in. kev_train_full.jsonl is train_split + holdout: now "
                         "that kev_dev.jsonl (22k rows, no overlap with train/holdout/test) is the validation "
                         "instrument, the holdout rows are free training data.")
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--out", default=str(COMP / "kev_train_ext.jsonl"))
    a = ap.parse_args()

    from datasets import load_dataset
    rng = random.Random(a.seed)
    out = []

    # 1) the competition's own train split (keeps the model in the competition's domain)
    comp_rows = [json.loads(l) for l in (COMP / a.comp_file).open(encoding="utf-8")]
    if a.comp:
        rng.shuffle(comp_rows)
        comp_rows = comp_rows[: a.comp]
    out += comp_rows * max(1, a.comp_repeat)
    print("competition %s: %d rows x%d" % (a.comp_file, len(comp_rows), max(1, a.comp_repeat)))

    # 2) XNLI train (the 14 non-English languages; English is left to MNLI below because XNLI's train split
    # IS MNLI -- XNLI-en rows carry the same premises, so mixing them would just duplicate English rows)
    for lang in XNLI_LANGS:
        n = a.xnli_en if lang == "en" else a.per_lang
        if not n:
            continue
        try:
            ds = load_dataset("facebook/xnli", lang, split="train")
        except Exception as exc:  # noqa: BLE001
            print("  xnli %s: SKIPPED (%s: %s)" % (lang, type(exc).__name__, str(exc)[:120]))
            continue
        rows = [{"premise": ds[i]["premise"], "hypothesis": ds[i]["hypothesis"], "label": ds[i]["label"]}
                for i in range(len(ds))]
        picked = balanced(rows, n, rng)
        out += [rec(r["premise"], r["hypothesis"], r["label"], "xnli", lang) for r in picked]
        print("  xnli %-2s: %d rows (of %d)" % (lang, len(picked), len(rows)))

    # 3) natural English MNLI (the same corpus, without the round-trip through translation)
    try:
        ds = load_dataset("nyu-mll/glue", "mnli", split="train")
        rows = [{"premise": ds[i]["premise"], "hypothesis": ds[i]["hypothesis"], "label": ds[i]["label"]}
                for i in range(len(ds))]
        picked = balanced(rows, a.mnli, rng)
        out += [rec(r["premise"], r["hypothesis"], r["label"], "mnli", "en") for r in picked]
        print("  mnli  en: %d rows (of %d)" % (len(picked), len(rows)))
    except Exception as exc:  # noqa: BLE001
        print("  mnli: SKIPPED (%s: %s)" % (type(exc).__name__, str(exc)[:120]))

    rng.shuffle(out)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print("wrote %d records -> %s" % (len(out), a.out))


if __name__ == "__main__":
    main()
