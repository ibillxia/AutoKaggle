"""Build *on-distribution* external data for contradictory-my-dear-watson.

Provenance measured on 2026-10-14 (exact (premise, hypothesis) match, 0 unmatched):

    train.csv 12120 = MNLI dev_matched 6870 + XNLI validation 1744 + XNLI test 3506
    test.csv   5195 = MNLI dev_matched 2945 + XNLI validation  746 + XNLI test 1504
    train.csv n test.csv = 0        MNLI dev_matched is 9815 and 100% consumed
    MNLI train / XNLI train splits -> 0 hits

So the competition drew English rows from MNLI dev_matched and non-English rows from a uniform
sample of XNLI validation u test. That is why the previous "external data" run -- which used the
XNLI/MNLI *train* splits, i.e. translations of MNLI **train** -- only bought +0.9 pt: wrong genre.
The pools the test was actually sampled from still have ~109k unused rows, and every one of them is
provably outside test.csv (the private test lives inside test.csv, which is fully matched).

This script therefore adds rows from the *same pools* the competition sampled:

    * MNLI `validation_mismatched`  (English; dev_matched is exhausted by the competition itself)
    * XNLI `test`                    (non-English; XNLI `validation` is left untouched as the instrument)

XNLI `validation` is deliberately **not** trained on: it is what `kev_dev.jsonl` evaluates on, so
the instrument stays row-clean and the measured gain is an honest read on the competition's own
non-English test rows (which come from the same val u test pool).

Per-language cap defaults to the competition's own per-language train count, so the added rows
preserve the competition's language mix instead of drowning English (57% of the metric) in XNLI.

    python prep_watson_pool.py --out data/comp/contradictory-my-dear-watson/kev_train_pool.jsonl
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
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


def key(p, h):
    return (p.strip(), h.strip())


def seen_keys(extra):
    """Everything the model must not be handed: the competition's train + test text, our old holdout,
    and every row of the evaluation instrument."""
    seen = set()
    for name in ("kev_train.jsonl", "kev_train_full.jsonl", "kev_holdout.jsonl"):
        f = COMP / name
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            body = r["state"][len("Premise: "):]
            p, h = body.split("\n\nHypothesis: ", 1)
            seen.add(key(p, h))
    for csv_name in ("train.csv", "test.csv"):
        for r in csv.DictReader((COMP / csv_name).open(encoding="utf-8")):
            seen.add(key(r["premise"], r["hypothesis"]))
    for f in extra:
        if not Path(f).exists():
            continue
        for line in Path(f).open(encoding="utf-8"):
            r = json.loads(line)
            body = r["state"][len("Premise: "):]
            p, h = body.split("\n\nHypothesis: ", 1)
            seen.add(key(p, h))
    return seen


def comp_per_lang():
    """The competition's own per-language train counts -- the mix the pool rows must respect."""
    counts = defaultdict(int)
    for r in csv.DictReader((COMP / "train.csv").open(encoding="utf-8")):
        counts[r["language"].strip().lower()] += 1
    abbr = {"english": "en", "arabic": "ar", "bulgarian": "bg", "german": "de", "greek": "el",
            "spanish": "es", "french": "fr", "hindi": "hi", "russian": "ru", "swahili": "sw",
            "thai": "th", "turkish": "tr", "urdu": "ur", "vietnamese": "vi", "chinese": "zh"}
    return {abbr[k]: v for k, v in counts.items()}


def comp_english_keys():
    """kev_train_full.jsonl carries no language column, so English has to come from train.csv itself."""
    keys = set()
    for r in csv.DictReader((COMP / "train.csv").open(encoding="utf-8")):
        if r["language"].strip().lower() == "english":
            keys.add(key(r["premise"], r["hypothesis"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", type=float, default=1.0,
                    help="pool rows per language = mult x the competition's train count for that language")
    ap.add_argument("--en-cap", type=int, default=0, help="hard cap on English pool rows (0 = no cap)")
    ap.add_argument("--en-repeat", type=int, default=0,
                    help="extra copies of the competition's English rows. Raising --mult inflates non-English "
                         "only (the English pool is capped by MNLI dev_mismatched), which drifts the mix away "
                         "from the competition's 57%% English; this puts it back.")
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--instrument", default=str(COMP / "kev_dev.jsonl"),
                    help="rows of the evaluation instrument; excluded from the pool")
    ap.add_argument("--out", default=str(COMP / "kev_train_pool.jsonl"))
    a = ap.parse_args()

    from datasets import load_dataset
    rng = random.Random(a.seed)

    seen = seen_keys([a.instrument])
    print("excluded keys (comp train/test + holdout + instrument): %d" % len(seen), flush=True)
    quota = comp_per_lang()
    print("competition train per language: %s" % dict(sorted(quota.items())), flush=True)

    out, meta = [], []

    # ---- English: MNLI dev_mismatched. dev_matched is 100% consumed by train.csv+test.csv. ----
    ds = load_dataset("nyu-mll/glue", "mnli", split="validation_mismatched")
    en_rows = [{"premise": ds[i]["premise"], "hypothesis": ds[i]["hypothesis"], "label": ds[i]["label"]}
               for i in range(len(ds))]
    rng.shuffle(en_rows)
    en_keep = [r for r in en_rows if key(r["premise"], r["hypothesis"]) not in seen]
    if a.en_cap:
        en_keep = en_keep[:a.en_cap]
    out += [rec(r["premise"], r["hypothesis"], r["label"], "mnli-devmm", "en") for r in en_keep]
    meta += [{"lang": "en", "src": "mnli-devmm"} for _ in en_keep]
    print("  en  mnli/validation_mismatched: %d free -> keep %d" % (len(en_rows), len(en_keep)), flush=True)

    # ---- non-English: XNLI test (validation stays clean as the instrument) ----
    for lang in XNLI_LANGS:
        if lang == "en":
            continue
        target = max(1, int(round(a.mult * quota.get(lang, 0))))
        ds = load_dataset("facebook/xnli", lang, split="test")
        rows = [{"premise": ds[i]["premise"], "hypothesis": ds[i]["hypothesis"], "label": ds[i]["label"]}
                for i in range(len(ds))]
        rng.shuffle(rows)
        free = [r for r in rows if key(r["premise"], r["hypothesis"]) not in seen]
        keep = free[:target]
        out += [rec(r["premise"], r["hypothesis"], r["label"], "xnli-test", lang) for r in keep]
        meta += [{"lang": lang, "src": "xnli-test"} for _ in keep]
        print("  %-2s xnli/test: target %-4d keep %-4d (pool %d, free %d)"
              % (lang, target, len(keep), len(rows), len(free)), flush=True)

    # ---- the competition's own labelled rows go on top (they are the target distribution) ----
    n_comp = 0
    comp_rows = [json.loads(line) for line in (COMP / "kev_train_full.jsonl").open(encoding="utf-8")]
    en_keys = comp_english_keys()
    en_comp = []
    for r in comp_rows:
        body = r["state"][len("Premise: "):]
        p, h = body.split("\n\nHypothesis: ", 1)
        if key(p, h) in en_keys:
            en_comp.append(r)
    out += comp_rows
    n_comp = len(comp_rows)
    for i in range(a.en_repeat):
        out += en_comp
        print("  + en-repeat %d/%d: %d English competition rows" % (i + 1, a.en_repeat, len(en_comp)), flush=True)
    print("  + competition train (kev_train_full): %d rows (%d English)" % (n_comp, len(en_comp)), flush=True)

    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    Path(a.out).with_suffix(".meta.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta), encoding="utf-8")
    by_lang = defaultdict(int)
    for m in meta:
        by_lang[m["lang"]] += 1
    print("wrote %d rows (%d pool + %d competition) -> %s" % (len(out), len(meta), n_comp, a.out), flush=True)
    print("pool mix: %s" % dict(sorted(by_lang.items())), flush=True)


if __name__ == "__main__":
    main()
