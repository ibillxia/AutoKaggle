"""Fine-tune a multilingual encoder (XLM-R large) for the 15-language NLI task.

Why this exists: the kev route -- a released decision head on top of Qwen3.5-0.8B -- plateaued at
COMP-MIX 0.8176/0.8244/0.8201 for delta/ext/pool, and the per-language ORACLE over all three members
is only 0.8269. English carries 56.7% of the metric and sits at 0.869, so the 0.8B head cannot reach
the 0.896 that the 12th place on the leaderboard implies. A proper multilingual encoder is the move.

Reads the same kev JSONL format the rest of `validation/` uses, so every existing instrument
(`kev_dev.jsonl`, `kev_holdout.jsonl`) and analysis script works unchanged.

    python train_hf_nli.py --model models/xlm-roberta-large \
        --data data/comp/contradictory-my-dear-watson/kev_train_pool.jsonl \
        --eval data/comp/contradictory-my-dear-watson/kev_dev.jsonl \
        --precision bf16 --out runs/xlmr-large-pool

`--eval` writes a `rows.json`-compatible dump (keys / label / p) so `lang_breakdown.py` and
`analysis_watson_dev.py` can read it directly.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CLASSES = ["entailment", "neutral", "contradiction"]
C2I = {c: i for i, c in enumerate(CLASSES)}


def parse_record(rec):
    body = rec["state"][len("Premise: "):]
    premise, hypothesis = body.split("\n\nHypothesis: ", 1)
    rel = rec["questions"]["relation"]
    label = rel["label"]
    if isinstance(label, int):
        return premise.strip(), hypothesis.strip(), int(label)
    return premise.strip(), hypothesis.strip(), C2I[label]


def load_jsonl(path):
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    return [parse_record(r) for r in rows]


def build_tokenizer(model_dir):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("tokenizer load failed (%s: %s); is tokenizer.json present?" % (type(exc).__name__, exc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local model dir or HF id")
    ap.add_argument("--data", default="", help="training jsonl (kev format)")
    ap.add_argument("--eval", default="", help="labelled jsonl to score after training")
    ap.add_argument("--eval-only", default="", help="load --out and only score --eval")
    ap.add_argument("--out", required=True, help="output dir for the fine-tuned model")
    ap.add_argument("--out-rows", default="", help="write a rows.json-compatible probability dump here")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--warmup-frac", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--optim", default="adamw", choices=["adamw", "adamw8bit"],
                    help="adamw8bit keeps the optimizer state 4x smaller -- needed to full-fine-tune "
                         "XLM-R large on an 8 GB card; Kaggle's 16 GB T4 takes plain adamw")
    ap.add_argument("--grad-checkpoint", type=int, default=0)
    ap.add_argument("--no-schedule", type=int, default=0,
                    help="hold the learning rate constant (diagnostics: separates the cosine "
                         "schedule from the data pipeline)")
    ap.add_argument("--probe-loss", type=int, default=0,
                    help="every 50 steps also report the loss on the first 8 rows plus the L2 drift of "
                         "a head weight and an encoder weight (diagnostics)")
    ap.add_argument("--fixed-batch", type=int, default=0,
                    help="reuse one batch instead of shuffling (diagnostics: separates the "
                         "DataLoader path from the optimiser)")
    ap.add_argument("--freeze-embeddings", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--save", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="cap training rows (smoke tests)")
    a = ap.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    dev = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"
    if a.precision == "bf16" and dev == "cuda" and not torch.cuda.is_bf16_supported():
        print("bf16 unsupported on this GPU -> fp16", flush=True)
        a.precision = "fp16"

    tok = build_tokenizer(a.eval_only or a.model)
    src = a.eval_only or a.model
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=len(CLASSES))
    if a.grad_checkpoint:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    if a.freeze_embeddings:
        frozen = 0
        for n, p in model.named_parameters():
            if n.startswith("roberta.embeddings") or n.startswith("xlm_roberta.embeddings"):
                p.requires_grad = False
                frozen += p.numel()
        print("froze %.1fM embedding params" % (frozen / 1e6), flush=True)
    model.to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print("model %s  params %.1fM  device=%s  precision=%s" % (src, n_par / 1e6, dev, a.precision), flush=True)

    class DS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            p, h, y = self.rows[i]
            return {"premise": p, "hypothesis": h, "labels": y}

    def collate(batch):
        enc = tok([b["premise"] for b in batch], [b["hypothesis"] for b in batch],
                  padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
        enc["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        return enc

    def autocast_ctx():
        if dev != "cuda" or a.precision == "fp32":
            return torch.autocast("cpu", enabled=False)
        return torch.autocast("cuda", dtype=torch.bfloat16 if a.precision == "bf16" else torch.float16)

    @torch.no_grad()
    def predict(rows):
        model.eval()
        out = []
        for i in range(0, len(rows), a.eval_batch):
            chunk = rows[i:i + a.eval_batch]
            enc = tok([r[0] for r in chunk], [r[1] for r in chunk], padding=True, truncation=True,
                      max_length=a.max_len, return_tensors="pt").to(dev)
            with autocast_ctx():
                logits = model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for (p, h, y), pr in zip(chunk, probs):
                out.append({"keys": CLASSES, "label": int(y), "p": [float(x) for x in pr]})
        return out

    if a.eval_only:
        rows = load_jsonl(a.eval)
        preds = predict(rows)
        acc = sum(1 for r in preds if max(range(3), key=lambda j: r["p"][j]) == r["label"]) / len(preds)
        nll = -sum(math.log(max(r["p"][r["label"]], 1e-12)) for r in preds) / len(preds)
        print("EVAL n=%d acc=%.4f nll=%.4f" % (len(preds), acc, nll), flush=True)
        if a.out_rows:
            Path(a.out_rows).write_text(json.dumps(preds), encoding="utf-8")
            print("wrote %s" % a.out_rows, flush=True)
        return

    train_rows = load_jsonl(a.data)
    if a.limit:
        train_rows = train_rows[:a.limit]
    print("train rows %d  labels %s" % (len(train_rows),
          {c: sum(1 for r in train_rows if r[2] == i) for i, c in enumerate(CLASSES)}), flush=True)
    if a.fixed_batch:
        one = next(iter(DataLoader(DS(train_rows[:a.batch]), batch_size=a.batch, shuffle=False,
                                   collate_fn=collate, drop_last=False)))
        dl = [one]
    else:
        dl = DataLoader(DS(train_rows), batch_size=a.batch, shuffle=True, collate_fn=collate, drop_last=False)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (n.endswith("bias") or "LayerNorm" in n or n.endswith("layer_norm.weight")) else decay).append(p)
    if a.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit([{"params": decay, "weight_decay": a.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98), eps=1e-6)
    else:
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98), eps=1e-6)

    steps_per_epoch = max(1, math.ceil(len(dl) / a.accum))
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    warm = max(1, int(total_steps * a.warmup_frac))

    def lr_at(step):
        if a.no_schedule:
            return a.lr
        if step < warm:
            return a.lr * (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return a.lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scaler = torch.amp.GradScaler("cuda", enabled=(a.precision == "fp16" and dev == "cuda"))
    model.train()

    _pl_batch = _pl_snap = _pl_names = None
    if a.probe_loss:
        _pl_batch = next(iter(DataLoader(DS(train_rows[:min(8, len(train_rows))]), batch_size=min(8, len(train_rows)),
                                         shuffle=False, collate_fn=collate, drop_last=False)))
        _pl_batch = {k: v.to(dev) for k, v in _pl_batch.items()}
        _pl_names = {}
        _pl_snap = {}
        for n, p in model.named_parameters():
            if n.endswith("classifier.out_proj.weight"):
                _pl_names["head"] = n
            if "encoder.layer.0.attention.self.query.weight" in n:
                _pl_names["encL0"] = n
        for k, n in _pl_names.items():
            _pl_snap[k] = dict(model.named_parameters())[n].detach().clone().float()

    def probe_report(step):
        named = dict(model.named_parameters())
        model.eval()
        with torch.no_grad(), autocast_ctx():
            pl = model(**_pl_batch).loss.item()
        model.train()
        drift = {k: round((named[v].detach().float() - _pl_snap[k]).norm().item(), 5)
                 for k, v in _pl_names.items()}
        print("      [probe] step %d  fixed8-loss %.4f  drift %s" % (step, pl, drift), flush=True)

    t0 = time.time()
    step = 0
    running, seen = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for ep in range(int(math.ceil(a.epochs))):
        for i, batch in enumerate(dl):
            batch = {k: v.to(dev) for k, v in batch.items()}
            with autocast_ctx():
                loss = model(**batch).loss / a.accum
            scaler.scale(loss).backward()
            running += loss.item() * a.accum
            seen += 1
            if (i + 1) % a.accum == 0 or (i + 1) == len(dl):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], a.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0 or step == total_steps:
                    el = time.time() - t0
                    print("  step %d/%d  loss %.4f  lr %.2e  %.2fs/step  elapsed %.1fmin"
                          % (step, total_steps, running / max(1, seen), lr_at(step), el / max(1, step), el / 60),
                          flush=True)
                    if a.probe_loss:
                        probe_report(step)
                    running, seen = 0.0, 0
                if total_steps and step >= total_steps:
                    break
        if total_steps and step >= total_steps:
            break

    print("trained %d steps in %.1f min" % (step, (time.time() - t0) / 60), flush=True)
    if a.save:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(a.out)
        tok.save_pretrained(a.out)
        Path(a.out, "train_meta.json").write_text(json.dumps(
            {k: v for k, v in vars(a).items()}, indent=2), encoding="utf-8")
        print("saved %s" % a.out, flush=True)

    if a.eval:
        rows = load_jsonl(a.eval)
        preds = predict(rows)
        acc = sum(1 for r in preds if max(range(3), key=lambda j: r["p"][j]) == r["label"]) / len(preds)
        nll = -sum(math.log(max(r["p"][r["label"]], 1e-12)) for r in preds) / len(preds)
        print("EVAL n=%d acc=%.4f nll=%.4f" % (len(preds), acc, nll), flush=True)
        if a.out_rows:
            Path(a.out_rows).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out_rows).write_text(json.dumps(preds), encoding="utf-8")
            print("wrote %s" % a.out_rows, flush=True)


if __name__ == "__main__":
    main()
