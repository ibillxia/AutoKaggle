"""Kaggle inference kernel: score test.csv with one or more fine-tuned XLM-R large checkpoints.

Training happens outside this notebook (the multi-hundred-thousand-row recipe takes hours on a T4
and would blow the 2 h competition limit), so this kernel is inference-only: locate the mounts,
average the members' softmax probabilities, write submission.csv. Nothing here touches the network.
"""
import csv
import glob
import json
import math
import os
import time
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASSES = ["entailment", "neutral", "contradiction"]
MAXLEN = __MAXLEN__
# [(subdirectory-or-name-fragment, temperature, weight), ...] -- resolved against /kaggle/input.
# The temperature is the one fitted on the dev instrument; probabilities are tempered before
# averaging so the kernel reproduces the dev-time ensemble recipe exactly.
MODEL_SPEC = __MODEL_SPEC__
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


ALL_MODELS = [d for d in (os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json",
                                                               recursive=True))
              if is_hf_model(d)]


def resolve(prefer):
    # exact directory basename first, then the longest matching fragment
    for d in ALL_MODELS:
        if os.path.basename(d) == prefer:
            return d
    hits = [d for d in ALL_MODELS if prefer in d]
    if hits:
        return max(hits, key=lambda d: len(os.path.basename(d)))
    raise SystemExit("no xlm-roberta checkpoint matching %r under /kaggle/input (have %s)"
                     % (prefer, [os.path.basename(d) for d in ALL_MODELS]))


COMP = find_competition_dir()
MEMBERS = [(resolve(name), float(t), float(w)) for name, t, w in MODEL_SPEC]
log("competition %s" % COMP)
for d, t, w in MEMBERS:
    log("member w=%.3f T=%.3f  %s" % (w, t, d))

test_rows = []
with open(os.path.join(COMP, "test.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        test_rows.append((r["id"], (r.get("premise") or "").strip(),
                          (r.get("hypothesis") or "").strip()))
log("test rows %d" % len(test_rows))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = torch.float16


def score(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=len(CLASSES))
    model.to(DEV).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(test_rows), BATCH):
            chunk = test_rows[i:i + BATCH]
            enc = tok([c[1] for c in chunk], [c[2] for c in chunk], padding=True,
                      truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
            with torch.autocast("cuda", dtype=AMP, enabled=(DEV == "cuda")):
                lg = model(**enc).logits
            out += torch.softmax(lg.float(), -1).cpu().numpy().tolist()
    del model
    torch.cuda.empty_cache()
    return out


def temper(p, T):
    lg = [math.log(max(x, 1e-12)) / T for x in p]
    m = max(lg)
    e = [math.exp(x - m) for x in lg]
    s = sum(e)
    return [x / s for x in e]


total_w = sum(w for _, _, w in MEMBERS)
probs = [[0.0, 0.0, 0.0] for _ in test_rows]
for d, T, w in MEMBERS:
    log("scoring %s (w=%.3f T=%.3f)" % (os.path.basename(d), w, T))
    p = score(d)
    for i, row in enumerate(p):
        for k, v in enumerate(temper(row, T)):
            probs[i][k] += w * v / total_w
    log("done %s" % os.path.basename(d))

preds = [int(np.argmax(p)) for p in probs]
with open(SUBMISSION, "w", encoding="utf-8", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["id", "prediction"])
    for (rid, _, _), y in zip(test_rows, preds):
        wr.writerow([rid, y])
log("wrote %s  %d rows  dist %s" % (SUBMISSION, len(preds), dict(Counter(preds))))

# keep the probabilities around: they let us ensemble with the kev members offline
with open(os.path.join(WORK, "xlmr_test_probs.jsonl"), "w", encoding="utf-8") as fh:
    for (rid, _, _), p in zip(test_rows, probs):
        fh.write(json.dumps({"id": rid, "p": p}) + "\n")
log("wrote probabilities")
