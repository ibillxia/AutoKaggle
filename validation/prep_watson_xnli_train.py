"""Build a large multilingual NLI training set from the locally cached facebook/xnli arrows.

The competition's train.csv is a 12k subsample of this same public training corpus; the cache holds
392,702 rows for every one of the 15 languages. Using more of the source *training* split is
ordinary practice and does not touch any test label. Overlap with the competition test set is nil
(XNLI test and MNLI dev_mismatched are disjoint from XNLI/MNLI train).
"""
import argparse
import glob
import json
import random
from pathlib import Path

import pyarrow as pa

CACHE = Path("C:/Users/deepg/.cache/huggingface/datasets/facebook___xnli")
ROOT = Path("E:/Bill/AI/AutoKaggle")
OUT_DIR = ROOT / "data/comp/contradictory-my-dear-watson"
LABELS = {"0": "entailment", "1": "neutral", "2": "contradiction",
          0: "entailment", 1: "neutral", 2: "contradiction"}


def read_train(lang):
    hits = glob.glob(str(CACHE / lang / "0.0.0" / "*" / "xnli-train.arrow"))
    if not hits:
        return None
    t = pa.ipc.open_stream(pa.memory_map(hits[0], "r")).read_all()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", type=int, default=120000)
    ap.add_argument("--other", type=int, default=8000)
    ap.add_argument("--out", default="kev_train_xnli.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    langs = sorted(p.name for p in CACHE.iterdir() if p.is_dir())
    total = 0
    path = OUT_DIR / a.out
    stats = {}
    with path.open("w", encoding="utf-8") as f:
        for lang in langs:
            t = read_train(lang)
            if t is None:
                print("skip", lang, flush=True)
                continue
            idx = list(range(t.num_rows))
            cap = a.en if lang == "en" else a.other
            if cap and cap < len(idx):
                idx = rng.sample(idx, cap)
            cols = {c: t.column(c).to_pylist() for c in t.column_names}
            n = 0
            for i in idx:
                lab = cols["label"][i]
                lab = LABELS[lab]
                rec = {"state": "Premise: %s\n\nHypothesis: %s" % (cols["premise"][i],
                                                                  cols["hypothesis"][i]),
                       "questions": {"relation": {"label": lab}},
                       "_lang": lang}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            stats[lang] = n
            total += n
            print("%-3s %6d" % (lang, n), flush=True)
    print("wrote %s  total %d" % (path, total))
    (OUT_DIR / (a.out + ".stats.json")).write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
