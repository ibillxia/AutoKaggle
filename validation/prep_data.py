"""Build labelled request files for the two Kaggle classification tasks.

Both datasets are the public mirrors of Kaggle datasets/competitions:
  * IMDB 50K Movie Reviews  (Kaggle: lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
      -> binary sentiment, emitted as a `noul` question (yes/no), Jev's binary decision type
  * AG News                  (Kaggle: amananandrai/ag-news-classification-dataset)
      -> 4-way topic, emitted as a `choice` question over 4 named options

Output: one JSON object per line in kev's labelled-request shape
({"state": ..., "questions": {qid: {type, instructions, criteria, label, src}}}).
"""
import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "0")

OUT = Path(__file__).resolve().parent / "data"

AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
AG_DESC = {
    "World": "International news, politics, conflicts, disasters",
    "Sports": "Sporting events, teams, players, results",
    "Business": "Companies, markets, earnings, the economy",
    "Sci/Tech": "Science, software, hardware, the internet",
}

IMDB_Q = "Is the sentiment of this movie review positive?"
AG_Q = "Which news topic does this article belong to?"
SMS_Q = "Is this SMS message spam?"
DISASTER_Q = "Does this tweet report a real disaster?"

NLI_Q = "What is the logical relationship between the premise and the hypothesis?"
NLI_LABELS = ["entailment", "neutral", "contradiction"]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records):5d} -> {path}")


def build_imdb(n, seed, max_chars):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb", split="test")
    rng = random.Random(seed)
    pos = [i for i, y in enumerate(ds["label"]) if y == 1]
    neg = [i for i, y in enumerate(ds["label"]) if y == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    idx = pos[: n // 2] + neg[: n // 2]
    rng.shuffle(idx)
    out = []
    for i in idx:
        text = ds[i]["text"].strip()
        out.append({
            "state": text[:max_chars],
            "questions": {
                "sentiment": {"type": "noul", "instructions": IMDB_Q,
                              "label": bool(ds[i]["label"]), "src": "imdb"},
            },
        })
    print(f"imdb   : {len(out)} rows, pos={sum(r['questions']['sentiment']['label'] for r in out)}")
    return out


def build_ag_news(n, seed):
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news", split="test")
    rng = random.Random(seed)
    by = {}
    for i, y in enumerate(ds["label"]):
        by.setdefault(int(y), []).append(i)
    per = n // len(AG_LABELS)
    idx = []
    for lab in range(len(AG_LABELS)):
        rows = by[lab]; rng.shuffle(rows); idx += rows[:per]
    rng.shuffle(idx)
    out = []
    for i in idx:
        r = ds[i]
        state = f"Title: {r['text'].split(' (')[0].strip()}\n\nArticle: {r['text']}"
        out.append({
            "state": state,
            "questions": {
                "topic": {"type": "choice", "instructions": AG_Q,
                          "criteria": {k: AG_DESC[k] for k in AG_LABELS},
                          "label": AG_LABELS[int(r["label"])], "src": "ag_news"},
            },
        })
    print(f"ag_news: {len(out)} rows, per-class={per}")
    return out


def build_sms(n, seed):
    """Kaggle's SMS Spam Collection (uciml/sms-spam-collection-dataset), UCI origin.

    Held-out probe: neither `sms_spam` nor its UCI source appears in kev's training-source list,
    so this is the one task here that measures zero-shot transfer rather than in-domain accuracy.
    """
    from datasets import load_dataset
    ds = load_dataset("ucirvine/sms_spam", split="train")
    rng = random.Random(seed)
    ham = [i for i, y in enumerate(ds["label"]) if y == 0]
    spam = [i for i, y in enumerate(ds["label"]) if y == 1]
    rng.shuffle(ham); rng.shuffle(spam)
    half = min(n // 2, len(spam))
    idx = ham[:half] + spam[:half]
    rng.shuffle(idx)
    out = []
    for i in idx:
        out.append({
            "state": ds[i]["sms"].strip(),
            "questions": {
                "spam": {"type": "noul", "instructions": SMS_Q,
                         "label": bool(ds[i]["label"]), "src": "sms_spam"},
            },
        })
    print(f"sms    : {len(out)} rows, spam={sum(r['questions']['spam']['label'] for r in out)}")
    return out


def build_disaster(n, seed):
    """Kaggle `nlp-getting-started` (Real or Not? NLP with Disaster Tweets), metric = F1.

    The raw CSV under data/raw/ is byte-identical to the competition's train.csv (987,712 B, 7613
    rows), pulled from a public mirror because the competition data itself needs a rules accept.
    We hold out a stratified `n` rows and keep the complement as the supervised baseline's train
    split, so both arms see exactly the same test rows.
    """
    import csv
    raw = OUT / "raw" / "disaster_train.csv"
    rows = list(csv.DictReader(raw.open(encoding="utf-8")))
    rng = random.Random(seed)
    pos = [r for r in rows if r["target"] == "1"]
    neg = [r for r in rows if r["target"] == "0"]
    rng.shuffle(pos); rng.shuffle(neg)
    hold = pos[: n // 2] + neg[: n // 2]
    rng.shuffle(hold)
    held = {r["id"] for r in hold}

    def rec(r):
        return {
            "state": r["text"].strip(),
            "questions": {"disaster": {"type": "noul", "instructions": DISASTER_Q,
                                       "label": r["target"] == "1", "src": "disaster_tweets"}},
        }

    test = [rec(r) for r in hold]
    train = [rec(r) for r in rows if r["id"] not in held]
    print(f"disaster: test={len(test)} (pos={sum(r['questions']['disaster']['label'] for r in test)})"
          f"  train={len(train)}")
    return test, train


def build_nli(n, seed):
    """Proxy probe for Kaggle `contradictory-my-dear-watson` (multilingual NLI, metric = accuracy).

    That competition's own data is Kaggle-only, so we use GLUE MNLI-matched (English, 3-way NLI,
    same label set and same Choice primitive) as a stand-in to estimate whether kev's pointer head
    can do NLI at all. English-only is a stated limitation: the real comp is 15 languages.
    """
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
    rng = random.Random(seed)
    by = {}
    for i, y in enumerate(ds["label"]):
        by.setdefault(int(y), []).append(i)
    per = n // 3
    idx = []
    for lab in range(3):
        rows = by[lab]; rng.shuffle(rows); idx += rows[:per]
    rng.shuffle(idx)
    out = []
    for i in idx:
        r = ds[i]
        out.append({
            "state": f"Premise: {r['premise'].strip()}\n\nHypothesis: {r['hypothesis'].strip()}",
            "questions": {
                "relation": {"type": "choice", "instructions": NLI_Q,
                             "criteria": {k: k for k in NLI_LABELS},
                             "label": NLI_LABELS[int(r["label"])], "src": "mnli"},
            },
        })
    print(f"nli    : {len(out)} rows, per-class={per}")
    return out


XNLI_LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]


def build_xnli(per_lang, seed):
    """Multilingual NLI probe (15 languages) — XNLI is the corpus behind Kaggle's
    `contradictory-my-dear-watson`, so this estimates the score that competition would produce.
    """
    from datasets import load_dataset
    rng = random.Random(seed)
    out = []
    for lang in XNLI_LANGS:
        ds = load_dataset("facebook/xnli", lang, split="validation")
        by = {}
        for i, y in enumerate(ds["label"]):
            by.setdefault(int(y), []).append(i)
        idx = []
        for lab in range(3):
            rows = by.get(lab, []); rng.shuffle(rows)
            idx += rows[: max(1, per_lang // 3)]
        rng.shuffle(idx)
        for i in idx:
            r = ds[i]
            out.append({
                "state": f"Premise: {r['premise'].strip()}\n\nHypothesis: {r['hypothesis'].strip()}",
                "questions": {
                    "relation": {"type": "choice", "instructions": NLI_Q,
                                 "criteria": {k: k for k in NLI_LABELS},
                                 "label": NLI_LABELS[int(r["label"])], "src": f"xnli-{lang}"},
                },
                "lang": lang,
            })
    print(f"xnli   : {len(out)} rows over {len(XNLI_LANGS)} languages, {per_lang}/lang")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="rows per task")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--imdb-max-chars", type=int, default=4000)
    ap.add_argument("--disaster-n", type=int, default=1000, help="held-out rows for nlp-getting-started")
    ap.add_argument("--nli-n", type=int, default=300, help="rows for the NLI proxy probe")
    ap.add_argument("--xnli-per-lang", type=int, default=21, help="rows per language for the XNLI probe")
    a = ap.parse_args()
    write_jsonl(OUT / "imdb_test.jsonl", build_imdb(a.n, a.seed, a.imdb_max_chars))
    write_jsonl(OUT / "ag_news_test.jsonl", build_ag_news(a.n, a.seed))
    write_jsonl(OUT / "sms_spam_test.jsonl", build_sms(200, a.seed))
    d_test, d_train = build_disaster(a.disaster_n, a.seed)
    write_jsonl(OUT / "disaster_test.jsonl", d_test)
    write_jsonl(OUT / "disaster_train.jsonl", d_train)
    write_jsonl(OUT / "nli_test.jsonl", build_nli(a.nli_n, a.seed))
    write_jsonl(OUT / "xnli_test.jsonl", build_xnli(a.xnli_per_lang, a.seed))
