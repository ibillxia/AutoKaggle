"""Fallback submission kernel: write submission.csv from precomputed ensemble predictions.

Used only when the GPU pool refuses to schedule the real inference kernel and the submission
quota would otherwise go to waste.  The predictions themselves come from the two fine-tuned
XLM-R large checkpoints (ibillxia/xlmr-watson-ft-big + ibillxia/xlmr-watson-ft-full, tempered
5:5 on the dev instrument); this kernel is deliberately trivial so it needs no accelerator.
"""
import csv
import glob
import os

WORK = "/kaggle/working"
SUBMISSION = os.path.join(WORK, "submission.csv")

cands = glob.glob("/kaggle/input/**/*.csv", recursive=True)
SKIP = {"sample_submission.csv", "train.csv", "test.csv"}
cands = [c for c in cands if os.path.basename(c) not in SKIP]
print("candidates", cands, flush=True)


def is_prediction_csv(path):
    with open(path, encoding="utf-8") as fh:
        head = fh.readline().strip().lower().replace(" ", "")
    return head in ("id,prediction", "id,label")


cands = [c for c in cands if is_prediction_csv(c)]
if not cands:
    raise SystemExit("no prediction csv under /kaggle/input")

src = max(cands, key=os.path.getsize)
print("using", src, flush=True)

with open(src, encoding="utf-8", newline="") as fh:
    rows = list(csv.reader(fh))
with open(SUBMISSION, "w", encoding="utf-8", newline="") as fh:
    csv.writer(fh).writerows(rows)
print("wrote %s  rows=%d" % (SUBMISSION, len(rows) - 1), flush=True)
