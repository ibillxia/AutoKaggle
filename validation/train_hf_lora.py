"""LoRA fine-tuning for the multilingual NLI task.

Full fine-tuning of xlm-roberta-large on an 8 GB card is both memory-bound and unstable (the
encoder degenerates to near-constant sentence vectors and cross-entropy sits at ln 3). LoRA keeps
the pretrained encoder intact -- the synthetic control task reaches loss 0.0000 in this regime --
while only ~4 M parameters receive gradients, which fits comfortably and trains in minutes.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLASSES = ["entailment", "neutral", "contradiction"]
C2I = {c: i for i, c in enumerate(CLASSES)}


def load_rows(path, limit=0):
    out = []
    for line in Path(path).open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        body = rec["state"][len("Premise: "):]
        p, h = body.split("\n\nHypothesis: ", 1)
        lab = rec["questions"]["relation"]["label"]
        out.append((p.strip(), h.strip(), int(lab) if isinstance(lab, int) else C2I[lab]))
        if limit and len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/xlm-roberta-large")
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-rows", default="")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--warmup-frac", type=float, default=0.06)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", default="query,key,value,dense")
    ap.add_argument("--base-dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=int, default=1)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=3)
    if a.base_dtype == "bf16" and dev == "cuda":
        model = model.to(torch.bfloat16)
    cfg = LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
        target_modules=[t.strip() for t in a.targets.split(",") if t.strip()],
        modules_to_save=["classifier"], bias="none", task_type="SEQ_CLS",
    )
    model = get_peft_model(model, cfg)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable %.2fM / total %.1fM" % (n_tr / 1e6,
          sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    model.to(dev)

    rows = load_rows(a.data, a.limit)
    t0 = time.time()
    ids, masks = [], []
    for p, h, _ in rows:
        e = tok(p, h, truncation=True, max_length=a.max_len, add_special_tokens=True)
        ids.append(torch.tensor(e["input_ids"], dtype=torch.long))
        masks.append(torch.tensor(e["attention_mask"], dtype=torch.long))
    labels = torch.tensor([r[2] for r in rows], dtype=torch.long)
    print("tokenised %d rows in %.1fs  lens p50=%d p95=%d max=%d"
          % (len(rows), time.time() - t0, int(np.percentile([len(i) for i in ids], 50)),
             int(np.percentile([len(i) for i in ids], 95)), max(len(i) for i in ids)), flush=True)

    def make_batch(idx):
        ii = [ids[i] for i in idx]
        mm = [masks[i] for i in idx]
        n = max(len(x) for x in ii)
        bi = torch.zeros(len(ii), n, dtype=torch.long)
        bm = torch.zeros(len(ii), n, dtype=torch.long)
        for r, (x, m) in enumerate(zip(ii, mm)):
            bi[r, :len(x)] = x
            bm[r, :len(m)] = m
        return {"input_ids": bi.to(dev), "attention_mask": bm.to(dev),
                "labels": labels[idx].to(dev)}

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.98), eps=1e-8,
                            weight_decay=a.wd)
    n = len(rows)
    steps_per_epoch = max(1, math.ceil(n / (a.batch * a.accum)))
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    warm = max(1, int(total_steps * a.warmup_frac))

    def lr_at(step):
        if step < warm:
            return a.lr * (step + 1) / warm
        return a.lr * max(0.0, 1.0 - (step - warm) / max(1, total_steps - warm))

    model.train()
    step = 0
    micro = 0
    run_loss, run_n = 0.0, 0
    t0 = time.time()
    rng = np.random.RandomState(a.seed)
    amp_dtype = torch.bfloat16 if a.base_dtype == "bf16" else None
    done = False

    for ep in range(int(math.ceil(a.epochs))):
        perm = rng.permutation(n)
        for s in range(0, n, a.batch):
            if done:
                break
            idx = perm[s:s + a.batch]
            batch = make_batch(idx)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(**batch)
                loss = out.loss / a.accum
            loss.backward()
            micro += 1
            run_loss += float(loss) * a.accum
            run_n += 1
            if micro % a.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(params, a.clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % a.log_every == 0 or step == total_steps:
                    el = time.time() - t0
                    print("  step %d/%d  loss %.4f  lr %.2e  %.2fs/step  elapsed %.1fmin"
                          % (step, total_steps, run_loss / max(1, run_n), lr_at(step),
                             el / max(1, step), el / 60), flush=True)
                    run_loss, run_n = 0.0, 0
                if step >= total_steps:
                    done = True
                    break
    print("trained %d steps in %.1f min" % (step, (time.time() - t0) / 60), flush=True)

    if a.save:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        merged = model.merge_and_unload()
        merged = merged.to(torch.float32)
        merged.save_pretrained(a.out)
        tok.save_pretrained(a.out)
        Path(a.out, "train_meta.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")
        print("saved %s" % a.out, flush=True)
    else:
        merged = model.merge_and_unload().to(torch.float32)

    if a.eval:
        merged.eval()
        erows = load_rows(a.eval)
        preds = []
        with torch.no_grad():
            for s in range(0, len(erows), a.eval_batch):
                ch = erows[s:s + a.eval_batch]
                e = tok([r[0] for r in ch], [r[1] for r in ch], padding=True, truncation=True,
                        max_length=a.max_len, return_tensors="pt").to(dev)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                    lg = merged(**e).logits
                pr = torch.softmax(lg.float(), -1).cpu().numpy()
                for r, p in zip(ch, pr):
                    preds.append({"keys": CLASSES, "label": int(r[2]),
                                  "p": [float(x) for x in p]})
        acc = sum(1 for r in preds if max(range(3), key=lambda j: r["p"][j]) == r["label"]) / len(preds)
        nll = -sum(math.log(max(r["p"][r["label"]], 1e-12)) for r in preds) / len(preds)
        print("EVAL n=%d acc=%.4f nll=%.4f" % (len(preds), acc, nll), flush=True)
        if a.out_rows:
            Path(a.out_rows).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out_rows).write_text(json.dumps(preds), encoding="utf-8")
            print("wrote %s" % a.out_rows, flush=True)


if __name__ == "__main__":
    main()
