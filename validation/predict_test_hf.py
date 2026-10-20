"""Score test.csv with a saved HF checkpoint and write {id, p} jsonl.

    python predict_test_hf.py --model runs/xlmr-full-ld --out submissions/xlmr_full_test_probs.jsonl

Used to build offline probability ensembles (and as a fallback when the Kaggle GPU
queue is congested: probs go up as a dataset, the kernel only has to write them out).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASSES = ["entailment", "neutral", "contradiction"]
COMP = Path("E:/Bill/AI/AutoKaggle/data/comp/contradictory-my-dear-watson")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", default=str(COMP / "test.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tta", type=int, default=0, help="1 = average probs over (p,h) and (h,p)")
    a = ap.parse_args()

    ids, prem, hyp = [], [], []
    with open(a.test, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            ids.append(r["id"])
            prem.append(r["premise"])
            hyp.append(r["hypothesis"])
            if a.limit and len(ids) >= a.limit:
                break
    print("rows %d" % len(ids), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.precision]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=3)
    model.to(dev).eval()

    probs = []
    with torch.no_grad():
        for s in range(0, len(ids), a.batch):
            e = tok(prem[s:s + a.batch], hyp[s:s + a.batch], padding=True, truncation=True,
                    max_length=a.max_len, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=amp, enabled=(dev == "cuda" and a.precision != "fp32")):
                lg = model(**e).logits
            pr = torch.softmax(lg.float(), -1)
            if a.tta:
                e2 = tok(hyp[s:s + a.batch], prem[s:s + a.batch], padding=True, truncation=True,
                         max_length=a.max_len, return_tensors="pt").to(dev)
                with torch.autocast("cuda", dtype=amp, enabled=(dev == "cuda" and a.precision != "fp32")):
                    lg2 = model(**e2).logits
                pr = 0.5 * (pr + torch.softmax(lg2.float(), -1))
            probs.append(pr.cpu().numpy())
            if (s // a.batch) % 20 == 0:
                print("  batch %d/%d" % (s // a.batch, (len(ids) + a.batch - 1) // a.batch), flush=True)
    p = np.concatenate(probs, 0)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for i, row in zip(ids, p):
            fh.write(json.dumps({"id": i, "keys": CLASSES, "p": [float(x) for x in row]},
                                ensure_ascii=False) + "\n")
    lab = p.argmax(1)
    print("wrote %s  n=%d  %s" % (out, len(ids),
          {c: int((lab == k).sum()) for k, c in enumerate(CLASSES)}), flush=True)


if __name__ == "__main__":
    main()
