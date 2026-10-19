"""Minimal, explicit fine-tuner for the multilingual NLI task.

`train_hf_nli.py` works when it replays one fixed batch and silently stops learning when it streams
shuffled batches, which is not a behaviour any healthy optimiser produces. Rather than keep guessing,
this is a from-scratch loop with every layer of indirection removed:

  * the whole corpus is tokenised ONCE into per-row tensors; batching then just gathers indices and
    pads. The tokenizer, Dataset and DataLoader are out of the inner loop entirely.
  * one parameter group, plain AdamW (or bitsandbytes 8-bit for an 8 GB card), no GradScaler --
    bf16 needs no loss scaling and the disabled scaler in the other script was pure indirection.
  * an optional linear warmup + linear decay, and the loss is logged every `--log-every` steps.

Reads the same kev JSONL and writes the same rows.json-compatible probability dump as
train_hf_nli.py, so lang_breakdown.py / analysis_watson_dev.py work unchanged.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="")
    ap.add_argument("--eval", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-rows", default="")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--warmup-frac", type=float, default=0.06)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--optim", default="adamw", choices=["adamw", "adamw8bit"])
    ap.add_argument("--grad-checkpoint", type=int, default=0)
    ap.add_argument("--freeze-encoder", type=int, default=0,
                    help="train only the classification head on frozen features (control for whether "
                         "the features separate the labels at all)")
    ap.add_argument("--head-warmup", type=int, default=0,
                    help="freeze the encoder for the first N steps so the (randomly initialised) head "
                         "converges first; the encoder then adapts on top of a meaningful head instead "
                         "of being thrashed by head noise from step 0")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--freeze-below", type=int, default=-1,
                    help="freeze encoder.layer.0..N-1 (and the embeddings); -1 disables")
    ap.add_argument("--layer-decay", type=float, default=0.0,
                    help="layer-wise learning-rate decay (Howard & Ruder): layer i gets "
                         "top_lr * decay**(23-i), embeddings decay**24, head keeps --head-lr. "
                         "0 disables. Recommended 0.85-0.9 for a 24-layer encoder.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save", type=int, default=1)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    prec = a.precision
    if prec == "auto":
        if dev == "cuda":
            prec = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            prec = "fp32"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(prec)
    use_amp = dev == "cuda" and amp_dtype is not None

    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=3)
    if a.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(dev)
    if a.freeze_below >= 0:
        import re as _re
        frozen = []
        for nm, p in model.named_parameters():
            if "embeddings" in nm:
                p.requires_grad = False
                frozen.append(nm)
                continue
            m = _re.search(r"encoder\.layer\.(\d+)\.", nm)
            if m and int(m.group(1)) < a.freeze_below:
                p.requires_grad = False
                frozen.append(nm)
        print("freeze-below %d: froze %d tensors" % (a.freeze_below, len(frozen)), flush=True)
    if a.freeze_encoder:
        n_fr = 0
        for n, p in model.named_parameters():
            if not n.startswith("classifier"):
                p.requires_grad = False
                n_fr += p.numel()
        print("froze encoder %.1fM params" % (n_fr / 1e6), flush=True)
    opt_params = [p for p in model.parameters() if p.requires_grad]

    def is_nodecay(nm):
        return nm.endswith("bias") or "LayerNorm" in nm or "layer_norm" in nm

    if a.layer_decay > 0:
        import re as _re
        buckets, kinds = {}, {}
        for nm, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if nm.startswith("classifier"):
                depth, lr, kind = 99, a.head_lr, "head"
            elif "embeddings" in nm:
                depth, lr, kind = -1, a.lr * a.layer_decay ** 24, "enc"
            else:
                m = _re.search(r"encoder\.layer\.(\d+)\.", nm)
                depth = int(m.group(1)) if m else -1
                lr = a.lr * a.layer_decay ** (23 - depth)
                kind = "enc"
            key = (depth, round(lr, 12), 0.0 if is_nodecay(nm) else a.wd)
            buckets.setdefault(key, []).append(p)
            kinds[key] = kind
        order = list(buckets)
        groups = [{"params": buckets[k], "lr": k[1], "weight_decay": k[2]} for k in order]
        group_kind = [kinds[k] for k in order]
        print("layer-decay %.2f -> %d param groups, lr range %.2e .. %.2e"
              % (a.layer_decay, len(groups), min(k[1] for k in order), max(k[1] for k in order)),
              flush=True)
    else:
        groups = [{"params": opt_params, "lr": a.lr, "weight_decay": a.wd}]
        group_kind = ["enc"]

    if a.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, lr=a.lr, betas=(0.9, 0.98), eps=1e-8)
    else:
        opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.98), eps=1e-8)
    base_lrs = [g["lr"] for g in opt.param_groups]

    def apply_lr(frac, head_only=False):
        for g, b, kd in zip(opt.param_groups, base_lrs, group_kind):
            if head_only:
                g["lr"] = a.head_lr if kd == "head" else 0.0
            else:
                g["lr"] = b * frac
    print("device=%s precision=%s optim=%s params %.1fM grad_ckpt=%d"
          % (dev, prec, a.optim, sum(p.numel() for p in model.parameters()) / 1e6, a.grad_checkpoint),
          flush=True)
    if prec == "fp16":
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = None

    # ---- tokenise the corpus once -------------------------------------------------
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
    print("labels %s" % {c: int((labels == i).sum()) for i, c in enumerate(CLASSES)}, flush=True)

    def make_batch(idx):
        ii = [ids[i] for i in idx]
        mm = [masks[i] for i in idx]
        maxlen = max(len(x) for x in ii)
        bi = torch.zeros(len(ii), maxlen, dtype=torch.long)
        bm = torch.zeros(len(ii), maxlen, dtype=torch.long)
        for r, (x, m) in enumerate(zip(ii, mm)):
            bi[r, :len(x)] = x
            bm[r, :len(m)] = m
        return {"input_ids": bi.to(dev), "attention_mask": bm.to(dev),
                "labels": labels[idx].to(dev)}

    n = len(rows)
    steps_per_epoch = max(1, math.ceil(n / a.batch))
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    warm = max(1, int(total_steps * a.warmup_frac))

    def lr_at(step):
        if step < warm:
            return a.lr * (step + 1) / warm
        return a.lr * max(0.0, 1.0 - (step - warm) / max(1, total_steps - warm))

    model.train()
    step = 0
    run_loss, run_n = 0.0, 0
    t0 = time.time()
    rng = np.random.RandomState(a.seed)
    done = False

    def set_encoder_trainable(flag):
        n = 0
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = bool(flag)
                if flag:
                    n += p.numel()
        return n

    if a.head_warmup:
        set_encoder_trainable(False)
        # optimiser only carries params that require grad; rebuild it is unnecessary because the
        # frozen ones simply receive no gradient, but keep the LR bookkeeping explicit below.

    for ep in range(int(math.ceil(a.epochs))):
        perm = rng.permutation(n)
        for s in range(0, n, a.batch):
            if done:
                break
            idx = perm[s:s + a.batch]
            batch = make_batch(idx)
            if a.head_warmup and step == a.head_warmup:
                set_encoder_trainable(True)
                print("  step %d: encoder unfrozen, switching to lr %.1e" % (step, a.lr), flush=True)
            if a.head_warmup and step < a.head_warmup:
                apply_lr(1.0, head_only=True)
            else:
                apply_lr(lr_at(step - a.head_warmup) / a.lr)
            ctx = torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp)
            with ctx:
                out = model(**batch)
                loss = out.loss
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(opt_params, a.clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(opt_params, a.clip)
                opt.step()
            run_loss += loss.item()
            run_n += 1
            step += 1
            if step % a.log_every == 0 or step == total_steps:
                el = time.time() - t0
                print("  step %d/%d  loss %.4f  lr %.2e  %.2fs/step  elapsed %.1fmin"
                      % (step, total_steps, run_loss / max(1, run_n), lr_at(step), el / step, el / 60),
                      flush=True)
                run_loss, run_n = 0.0, 0
            if step >= total_steps:
                done = True
    print("trained %d steps in %.1f min" % (step, (time.time() - t0) / 60), flush=True)

    if a.save:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(a.out)
        tok.save_pretrained(a.out)
        Path(a.out, "train_meta.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")
        print("saved %s" % a.out, flush=True)

    if a.eval:
        erows = load_rows(a.eval)
        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(erows), a.eval_batch):
                ch = erows[s:s + a.eval_batch]
                e = tok([r[0] for r in ch], [r[1] for r in ch], padding=True, truncation=True,
                        max_length=a.max_len, return_tensors="pt").to(dev)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    lg = model(**e).logits
                pr = torch.softmax(lg.float(), -1).cpu().numpy()
                for r, p in zip(ch, pr):
                    preds.append({"keys": CLASSES, "label": int(r[2]), "p": [float(x) for x in p]})
        acc = sum(1 for r in preds if max(range(3), key=lambda j: r["p"][j]) == r["label"]) / len(preds)
        nll = -sum(math.log(max(r["p"][r["label"]], 1e-12)) for r in preds) / len(preds)
        print("EVAL n=%d acc=%.4f nll=%.4f" % (len(preds), acc, nll), flush=True)
        if a.out_rows:
            Path(a.out_rows).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out_rows).write_text(json.dumps(preds), encoding="utf-8")
            print("wrote %s" % a.out_rows, flush=True)


if __name__ == "__main__":
    main()
