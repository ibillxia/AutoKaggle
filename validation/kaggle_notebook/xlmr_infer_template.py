"""Kaggle inference kernel: load the fine-tuned XLM-R large from our dataset and score test.csv.

Training happens outside this notebook (the 255k-row recipe takes ~1.2 h on the local 4060 Ti but
would blow the 2 h competition limit on a T4), so this kernel is inference-only: find the mounts,
predict, write submission.csv. Nothing here touches the network.
"""
import csv
import glob
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASSES = ["entailment", "neutral", "contradiction"]
MAXLEN = __MAXLEN__
PREFER = "__MODEL_SUBDIR__"
BATCH = 64

WORK = "/kaggle/working"
SUBMISSION = os.path.join(WORK, "submission.csv")
t_start = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - t_start, msg), flush=True)


def find_competition_dir():
    for sub in glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True):
        d = os.path.dirname(sub)
        if os.path.isfile(os.path.join(d, "test.csv")):
            return d
    raise SystemExit("competition data not found under /kaggle/input")


def is_hf_model(d):
    if not os.path.isfile(os.path.join(d, "config.json")):
        return False
    if not (os.path.isfile(os.path.join(d, "model.safetensors"))
            or os.path.isfile(os.path.join(d, "pytorch_model.bin"))):
        return False
    try:
        meta = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return meta.get("model_type") == "xlm-roberta"


def find_model_dir():
    cands = [os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json", recursive=True)]
    cands = [d for d in cands if is_hf_model(d)]
    if not cands:
        raise SystemExit("no xlm-roberta checkpoint under /kaggle/input")
    for d in cands:
        if PREFER and (os.path.basename(d) == PREFER or PREFER in d):
            return d
    return cands[0]


COMP = find_competition_dir()
MODEL = find_model_dir()
log("competition %s" % COMP)
log("model       %s" % MODEL)

test_rows = []
with open(os.path.join(COMP, "test.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        test_rows.append((r["id"], (r.get("premise") or "").strip(),
                          (r.get("hypothesis") or "").strip()))
log("test rows %d" % len(test_rows))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(CLASSES))
model.to(DEV).eval()
AMP = torch.float16
log("params %.1fM on %s" % (sum(p.numel() for p in model.parameters()) / 1e6, DEV))

probs = []
with torch.no_grad():
    for i in range(0, len(test_rows), BATCH):
        chunk = test_rows[i:i + BATCH]
        enc = tok([c[1] for c in chunk], [c[2] for c in chunk], padding=True, truncation=True,
                  max_length=MAXLEN, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=AMP, enabled=(DEV == "cuda")):
            lg = model(**enc).logits
        probs += torch.softmax(lg.float(), -1).cpu().numpy().tolist()
        if (i // BATCH) % 20 == 0:
            log("scored %d/%d" % (min(i + BATCH, len(test_rows)), len(test_rows)))

preds = [int(np.argmax(p)) for p in probs]
with open(SUBMISSION, "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "prediction"])
    for (rid, _, _), y in zip(test_rows, preds):
        w.writerow([rid, y])

from collections import Counter  # noqa: E402
log("wrote %s  %d rows  dist %s" % (SUBMISSION, len(preds), dict(Counter(preds))))

# keep the probabilities around: they let us ensemble with the kev members offline
with open(os.path.join(WORK, "xlmr_test_probs.jsonl"), "w", encoding="utf-8") as fh:
    for (rid, _, _), p in zip(test_rows, probs):
        fh.write(json.dumps({"id": rid, "p": p}) + "\n")
log("wrote probabilities")
