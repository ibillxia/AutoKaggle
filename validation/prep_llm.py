"""Turn llm-classification-finetuning's train.csv into kev requests.

The competition ships a real `test.csv` of 3 rows only -- the scored split is hidden and mounted into the
notebook at run time -- so every local number here comes from a stratified slice of train.csv.

A record is a `choice` question over the three winner columns:

    state  Prompt: <prompt>\n\nResponse A: <a>\n\nResponse B: <b>
    Q      Which response is better?   criteria = winner_model_a / winner_model_b / winner_tie

Fields are truncated to the same widths the notebook uses (see validation/build_notebook.py), because
that prefix is all the model ever sees.

    python prep_llm.py [--holdout 1500] [--train-cap 12000]
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMP = HERE.parent / "data" / "comp" / "llm-classification-finetuning"

PROMPT_CHARS, RESP_CHARS = 300, 600
CRITERIA = ["winner_model_a", "winner_model_b", "winner_tie"]
INSTRUCTIONS = "Which response is better?"


def text_of(v, width):
    """Arena cells are JSON arrays of strings (multi-turn); join them so the model sees prose, not brackets."""
    try:
        x = json.loads(v)
        if isinstance(x, list):
            return " ".join(str(i) for i in x)[:width]
        return str(x)[:width]
    except Exception:  # noqa: BLE001
        return (v or "")[:width]


def state_of(r):
    return "Prompt: %s\n\nResponse A: %s\n\nResponse B: %s" % (
        text_of(r.get("prompt"), PROMPT_CHARS),
        text_of(r.get("response_a"), RESP_CHARS),
        text_of(r.get("response_b"), RESP_CHARS),
    )


def record(r, labelled=True):
    q = {"type": "choice", "instructions": INSTRUCTIONS,
         "criteria": {c: c for c in CRITERIA}, "src": "llm-classification-finetuning"}
    if labelled:
        q["label"] = next(c for c in CRITERIA if r[c] == "1")
    return {"state": state_of(r), "questions": {"winner": q}}


def main():
    global PROMPT_CHARS, RESP_CHARS
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=1500)
    ap.add_argument("--train-cap", type=int, default=12000, help="cap the fine-tune split (long states are slow)")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--prompt-chars", type=int, default=300)
    ap.add_argument("--resp-chars", type=int, default=600)
    ap.add_argument("--tag", default="", help="suffix for the written files, e.g. _s")
    a = ap.parse_args()

    PROMPT_CHARS, RESP_CHARS = a.prompt_chars, a.resp_chars
    print("state widths: prompt %d + 2 x response %d = %d chars"
          % (PROMPT_CHARS, RESP_CHARS, PROMPT_CHARS + 2 * RESP_CHARS))

    rows = list(csv.DictReader((COMP / "train.csv").open(encoding="utf-8")))
    print("train rows", len(rows))

    by = defaultdict(list)
    for r in rows:
        by[next(c for c in CRITERIA if r[c] == "1")].append(r)
    rng = random.Random(a.seed)
    held = []
    for k, rs in by.items():
        rng.shuffle(rs)
        held += rs[: int(round(a.holdout * len(rs) / len(rows)))]
    held_ids = {r["id"] for r in held}
    pool = [r for r in rows if r["id"] not in held_ids]
    rng.shuffle(pool)
    fit = pool[: a.train_cap]
    print("holdout", len(held), "| fit", len(fit), "of", len(pool))

    def dump(name, recs):
        p = COMP / name
        with p.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(recs):6d} -> {p.name}")

    dump("kev_holdout%s.jsonl" % a.tag, [record(r) for r in held])
    dump("kev_train%s.jsonl" % a.tag, [record(r) for r in fit])
    (COMP / ("holdout_ids%s.json" % a.tag)).write_text(json.dumps([r["id"] for r in held]), encoding="utf-8")


if __name__ == "__main__":
    main()
