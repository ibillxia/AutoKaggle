"""Diagnose why the XLM-R fine-tune loss sits at ln(3) forever.

Runs a handful of identical steps under different (grad-checkpoint, optimiser) pairs and reports the
gradient norm per parameter group. If the norm of the encoder block params is 0 or NaN, the head is the
only thing learning and the run is broken.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASSES = ["entailment", "neutral", "contradiction"]
ROOT = Path(__file__).resolve().parent.parent


def load_rows(path, limit):
    rows = []
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        body = rec["state"][len("Premise: "):]
        p, h = body.split("\n\nHypothesis: ", 1)
        lab = rec["questions"]["relation"]["label"]
        rows.append((p.strip(), h.strip(), int(lab) if isinstance(lab, int) else CLASSES.index(lab)))
        if limit and len(rows) >= limit:
            break
    return rows


def run(tag, rows, tok, model_dir, gc, optim_name, steps=20, batch=8, accum=2, lr=1e-4):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    if gc:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to("cuda")
    model.train()

    class DS(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            p, h, y = rows[i]
            return {"p": p, "h": h, "y": y}

    def collate(b):
        e = tok([x["p"] for x in b], [x["h"] for x in b], padding=True, truncation=True,
                max_length=256, return_tensors="pt")
        e["labels"] = torch.tensor([x["y"] for x in b])
        return e

    dl = DataLoader(DS(), batch_size=batch, shuffle=True, collate_fn=collate)
    params = [p for p in model.parameters() if p.requires_grad]
    if optim_name == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=lr)
    else:
        opt = torch.optim.AdamW(params, lr=lr)

    probe = {}
    for n, p in model.named_parameters():
        if n.endswith("classifier.out_proj.weight"):
            probe["head"] = n
        if "encoder.layer.0.attention.self.query.weight" in n:
            probe["blk0"] = n
        if "encoder.layer.23" in n and n.endswith("attention.output.dense.weight"):
            probe["blk23"] = n
    named = dict(model.named_parameters())

    opt.zero_grad(set_to_none=True)
    losses, gnorms = [], {}
    i = 0
    for step in range(steps):
        for _ in range(accum):
            b = next(iter(dl)) if False else None
        # rebuild iterator each step so the loop stays simple
        batches = list(dl)
        for bi in range(accum):
            batch = {k: v.to("cuda") for k, v in batches[i % len(batches)].items()}
            i += 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / accum
            loss.backward()
            losses.append(loss.item() * accum)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        gnorms[step] = float(gn)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step in (0, 1, 4, 9, 19):
            snap = {k: float(named[v].grad.norm()) if named[v].grad is not None else None
                    for k, v in probe.items()}
            print("  [%s] step %2d loss %.4f gnorm_total %.4f  %s"
                  % (tag, step, losses[-1], gnorms[step], snap), flush=True)
    print("  [%s] loss %.4f -> %.4f  (mean first5 %.4f, mean last5 %.4f)"
          % (tag, losses[0], losses[-1], sum(losses[:5]) / 5, sum(losses[-5:]) / 5), flush=True)
    del model, opt, params
    torch.cuda.empty_cache()
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/xlm-roberta-large")
    ap.add_argument("--data", default="data/comp/contradictory-my-dear-watson/kev_train_pool.jsonl")
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    rows = load_rows(ROOT / a.data, a.limit)
    tok = AutoTokenizer.from_pretrained(ROOT / a.model)
    print("rows %d" % len(rows), flush=True)
    combos = [(0, "adamw"), (1, "adamw"), (1, "adamw8bit")]
    if False:
        pass
    for gc, opt_name in combos:
        t = time.time()
        run("gc=%d/%s" % (gc, opt_name), rows, tok, str(ROOT / a.model), gc, opt_name, steps=a.steps)
        print("  (%.0fs)" % (time.time() - t), flush=True)


if __name__ == "__main__":
    main()
