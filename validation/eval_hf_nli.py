"""Score a saved HF checkpoint on the watson dev instrument and write rows.json for analysis.

    python eval_hf_nli.py --model runs/xlmr-fb12-huge --out validation/runs/xlmr-huge-dev/rows.json

The output format matches train_hf_nli2.py so analysis_watson_dev.py can consume it directly.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASSES = ["entailment", "neutral", "contradiction"]
L2I = {c: i for i, c in enumerate(CLASSES)}
ROOT = Path("E:/Bill/AI/AutoKaggle")
DEV_JSONL = ROOT / "data/comp/contradictory-my-dear-watson/kev_dev.jsonl"


def load_rows(path, limit=0):
    prem, hyp, lab = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            state = r.get("state", "")
            if "Hypothesis:" in state:
                p, h = state.split("Hypothesis:", 1)
                p = p.replace("Premise:", "").strip()
                h = h.strip()
            else:
                p, h = state, ""
            prem.append(p)
            hyp.append(h)
            lab.append(L2I[r["questions"]["relation"]["label"]])
            if limit and len(lab) >= limit:
                break
    return prem, hyp, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", default=str(DEV_JSONL))
    ap.add_argument("--out", default="")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tta", type=int, default=0, help="1 = average probs over (p,h) and (h,p)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.precision]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=3)
    model.to(dev).eval()

    prem, hyp, lab = load_rows(a.eval, a.limit)
    print("rows %d" % len(lab), flush=True)
    preds = []
    with torch.no_grad():
        for s in range(0, len(lab), a.batch):
            pp, hh = prem[s:s + a.batch], hyp[s:s + a.batch]
            e = tok(pp, hh, padding=True, truncation=True,
                    max_length=a.max_len, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=amp, enabled=(dev == "cuda" and a.precision != "fp32")):
                lg = model(**e).logits
            pr = torch.softmax(lg.float(), -1)
            if a.tta:
                e2 = tok(hh, pp, padding=True, truncation=True,
                         max_length=a.max_len, return_tensors="pt").to(dev)
                with torch.autocast("cuda", dtype=amp, enabled=(dev == "cuda" and a.precision != "fp32")):
                    lg2 = model(**e2).logits
                pr = 0.5 * (pr + torch.softmax(lg2.float(), -1))
            pr = pr.cpu().numpy()
            for k, p in enumerate(pr):
                preds.append({"keys": CLASSES, "label": int(lab[s + k]),
                              "p": [float(x) for x in p]})
    acc = float(np.mean([int(np.argmax(r["p"])) == r["label"] for r in preds]))
    nll = -float(np.mean([math.log(max(r["p"][r["label"]], 1e-12)) for r in preds]))
    print("EVAL n=%d acc=%.4f nll=%.4f" % (len(preds), acc, nll), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(preds), encoding="utf-8")
        print("wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
