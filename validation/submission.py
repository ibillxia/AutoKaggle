"""Build real Kaggle submissions for the three competitions kev suits.

Two modes:

  prep     read the competition's own csv -> kev request jsonl
           (labelled `kev_train.jsonl` for fine-tuning, unlabelled `kev_test.jsonl` to predict,
           plus `test_ids.json` to map predictions back onto submission rows)

  predict  run a kev checkpoint over `kev_test.jsonl` -> submission.csv

    python submission.py prep    --comp nlp-getting-started [--holdout 800] [--seed 20261013]
    python submission.py predict --comp nlp-getting-started --run ../runs/kev-0.8b-ft-disaster \\
                                 --out ../submissions/kev-0.8b-ft.csv

The competitions:
  * nlp-getting-started          (Real or Not? NLP with Disaster Tweets)  metric = F1
        noul question over `text`  ->  target 0/1
  * contradictory-my-dear-watson (Contradictory, My Dear Watson)          metric = accuracy
        choice question over premise+hypothesis -> prediction 0/1/2 (0=entailment 1=neutral 2=contradiction)
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_wincompat"))                 # fcntl / resource shims (Windows)
sys.path.insert(0, "E:/Bill/AI/AutoKaggle/third_party/kev")

DATA = HERE.parent / "data" / "comp"            # real competition csvs (downloaded via kaggle CLI)

# label order matches GLUE MNLI / XNLI / this competition's own encoding
CMDW_LABELS = ["entailment", "neutral", "contradiction"]

COMPS = {
    "nlp-getting-started": {
        "train": "train.csv", "test": "test.csv",
        "question": "disaster",
        "instructions": "Does this tweet report a real disaster?",
        "type": "noul",
        "sub_id": "id", "sub_col": "target",
    },
    "contradictory-my-dear-watson": {
        "train": "train.csv", "test": "test.csv",
        "question": "relation",
        "instructions": "What is the logical relationship between the premise and the hypothesis?",
        "type": "choice",
        "sub_id": "id", "sub_col": "prediction",
    },
}


def rows_of(comp, split):
    path = DATA / comp / COMPS[comp][f"{'train' if split == 'train' else 'test'}"]
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_record(comp, r, labelled):
    cfg = COMPS[comp]
    if comp == "nlp-getting-started":
        state = (r["text"] or "").strip()
        q = {"type": "noul", "instructions": cfg["instructions"]}
        if labelled:
            q["label"] = r["target"] == "1"
    else:
        state = f"Premise: {(r['premise'] or '').strip()}\n\nHypothesis: {(r['hypothesis'] or '').strip()}"
        q = {"type": "choice", "instructions": cfg["instructions"],
             "criteria": {k: k for k in CMDW_LABELS}}
        if labelled:
            q["label"] = CMDW_LABELS[int(r["label"])]
    q["src"] = comp
    return {"state": state, "questions": {cfg["question"]: q}}


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records):6d} -> {path}")


def prep(a):
    cfg = COMPS[a.comp]
    train = rows_of(a.comp, "train")
    test = rows_of(a.comp, "test")

    if a.holdout and a.holdout < len(train):
        import random
        rng = random.Random(a.seed)
        # stratified holdout so the held-out slice keeps the class balance
        by = {}
        for r in train:
            by.setdefault(r.get("target", r.get("label")), []).append(r)
        held = []
        for k, rs in by.items():
            rng.shuffle(rs)
            held += rs[: max(1, int(round(a.holdout * len(rs) / len(train))))]
        rng.shuffle(held)
        held_ids = {r["id"] for r in held}
        ft_rows = [r for r in train if r["id"] not in held_ids]
        print(f"[prep] {a.comp}: held out {len(held)}, fine-tune on {len(ft_rows)}")
        write_jsonl(DATA / a.comp / "kev_holdout.jsonl", [to_record(a.comp, r, True) for r in held])
    else:
        ft_rows = train
        print(f"[prep] {a.comp}: fine-tune on all {len(ft_rows)} rows (no holdout)")

    write_jsonl(DATA / a.comp / "kev_train.jsonl", [to_record(a.comp, r, True) for r in ft_rows])
    write_jsonl(DATA / a.comp / "kev_test.jsonl", [to_record(a.comp, r, False) for r in test])
    (DATA / a.comp / "test_ids.json").write_text(
        json.dumps([r[cfg["sub_id"]] for r in test]), encoding="utf-8")
    print(f"[prep] test rows = {len(test)}, submission column = {cfg['sub_col']}")


def predict(a):
    import torch
    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    from kev.suite import SERVING_CONTEXT

    cfg = COMPS[a.comp]
    # kev.data.load_records / materialize demand a label on every question (it is only used for
    # metrics), so the unlabelled test split is read straight off disk and each question gets a
    # placeholder label -- the scored probabilities do not depend on it.
    placeholder = False if comp_key(cfg) else CMDW_LABELS[0]
    with (DATA / a.comp / "kev_test.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    for rec in records:
        for q in rec["questions"].values():
            q.setdefault("label", placeholder)
    ids = json.loads((DATA / a.comp / "test_ids.json").read_text(encoding="utf-8"))
    assert len(ids) == len(records), f"id/record mismatch: {len(ids)} vs {len(records)}"

    opts = LoadOptions(dtype=torch.bfloat16 if a.dtype == "bf16" else None, backend="torch")
    t0 = time.time()
    predictor = LocalPredictor(a.run, a.device, opts, context=SERVING_CONTEXT)
    print(f"[{Path(a.run).name} / {a.comp}] loaded in {time.time() - t0:.1f}s "
          f"(T={predictor.checkpoint.meta.temperature:.4f})", flush=True)

    qid = cfg["question"]
    out, prob_rows, fallback, t1 = [], [], 0, time.time()
    for i, rec in enumerate(records):
        try:
            probs = predictor(rec)["probabilities"][qid]
            key = max(probs, key=probs.get)
            if comp_key(cfg):                      # noul: "true"/"false" -> 1/0
                value = 1 if key == "true" else 0
            else:                                  # choice: label string -> competition code
                value = CMDW_LABELS.index(key)
        except Exception as e:                     # never drop a submission row
            fallback += 1
            probs, value = {}, a.fallback
            print(f"  ! row {i} ({ids[i]}) fell back to {value}: {type(e).__name__}: {e}", flush=True)
        out.append((ids[i], value))
        prob_rows.append({"id": ids[i], "pred": value, "p": probs})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(records)}  {(time.time() - t1) / (i + 1):.3f} s/row", flush=True)

    wall = time.time() - t1
    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([cfg["sub_id"], cfg["sub_col"]])
        w.writerows(out)
    with target.with_suffix(".probs.jsonl").open("w", encoding="utf-8") as f:
        for row in prob_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter
    dist = Counter(v for _, v in out)
    print(f"[predict] {len(out)} rows in {wall:.0f}s ({wall / len(out):.3f} s/row), "
          f"fallbacks={fallback}")
    print(f"[predict] label distribution: {dict(sorted(dist.items()))}")
    if a.device == "cuda":
        print(f"[predict] peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"[predict] wrote {target}")


def comp_key(cfg):
    """True when the question is a noul (yes/no) question -> keys are 'true'/'false'."""
    return cfg["type"] == "noul"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "predict"])
    ap.add_argument("--comp", required=True, choices=sorted(COMPS))
    ap.add_argument("--holdout", type=int, default=0, help="stratified holdout size for local validation")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--run", help="kev checkpoint directory (predict)")
    ap.add_argument("--out", default="", help="submission csv path (predict)")
    ap.add_argument("--device", default=None, help="cuda / cpu (predict)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--fallback", type=int, default=0, help="label used when a row cannot be scored")
    a = ap.parse_args()
    if a.mode == "prep":
        prep(a)
    else:
        if not a.run:
            ap.error("--run is required for predict")
        if a.device is None:
            import torch
            a.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not a.out:
            a.out = str(HERE.parent / "submissions" / f"{Path(a.run).name}__{a.comp}.csv")
        predict(a)


if __name__ == "__main__":
    main()
