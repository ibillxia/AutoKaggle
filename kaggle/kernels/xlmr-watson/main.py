"""Kaggle kernel body: fine-tune XLM-R large on the watson NLI task, offline.

`validation/build_xlmr_notebook.py` substitutes the __PLACEHOLDER__s and writes this next to a
kernel-metadata.json, then pushes it. Code competitions run with the internet off, so everything it
needs is mounted:

  the competition's own split   (competition_sources) -> test.csv / train.csv / sample_submission.csv
  our dataset (dataset_sources) -> xlm-roberta-large/ + kev_train_pool*.jsonl

Mount layout is globbed rather than guessed: Kaggle auto-extracts uploaded archives, so the inner
folder name is not something to hard-code.
"""
import csv
import glob
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASSES = ["entailment", "neutral", "contradiction"]
C2I = {c: i for i, c in enumerate(CLASSES)}

EPOCHS = 3.0
LR = 5e-05
BATCH = 16
ACCUM = 1
MAXLEN = 256
GRAD_CKPT = True        # trade compute for VRAM: a 16 GB T4 is tight for XLM-R large
FREEZE_BELOW = 18  # freeze embeddings + encoder layers 0..N-1
USE_POOL = True
EN_REPEAT = 0        # extra copies of the competition's English rows (mix repair for a big pool)
SEED = 0
MODEL_DIR_OVERRIDE = """"""
POOL_NAME = "kev_pool_extra.jsonl"

WORK = "/kaggle/working"
SUBMISSION = os.path.join(WORK, "submission.csv")
t_start = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - t_start, msg), flush=True)


# ------------------------------------------------------------------ locate the mounts
def find_competition_dir():
    for sub in glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True):
        d = os.path.dirname(sub)
        if os.path.isfile(os.path.join(d, "test.csv")) and os.path.isfile(os.path.join(d, "train.csv")):
            return d
    for sub in glob.glob("/kaggle/input/**/test.csv", recursive=True):
        d = os.path.dirname(sub)
        if os.path.isfile(os.path.join(d, "train.csv")):
            return d
    raise SystemExit("competition data not found under /kaggle/input")


def find_model_dir():
    if MODEL_DIR_OVERRIDE and os.path.isdir(MODEL_DIR_OVERRIDE):
        return MODEL_DIR_OVERRIDE
    for cfg in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(cfg)
        try:
            meta = json.load(open(cfg, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if meta.get("model_type") == "xlm-roberta" and (
                os.path.isfile(os.path.join(d, "model.safetensors")) or os.path.isfile(os.path.join(d, "pytorch_model.bin"))):
            return d
    raise SystemExit("xlm-roberta weights not found under /kaggle/input")


def find_pool():
    if not USE_POOL:
        return ""
    pats = ["/kaggle/input/**/kev_train_pool*.jsonl"]
    if POOL_NAME:
        pats.insert(0, "/kaggle/input/**/%s" % POOL_NAME)
    hits = []
    for pat in pats:
        hits += [h for h in glob.glob(pat, recursive=True) if os.path.isfile(h)]
    return sorted(set(hits))[0] if hits else ""


COMP_DIR = find_competition_dir()
MODEL_DIR = find_model_dir()
POOL = find_pool()
log("competition dir %s" % COMP_DIR)
log("model dir       %s" % MODEL_DIR)
log("pool            %s" % (POOL or "(none -- competition train only)"))


# ------------------------------------------------------------------ data
def read_comp_train():
    rows = []
    with open(os.path.join(COMP_DIR, "train.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(((r["premise"] or "").strip(), (r["hypothesis"] or "").strip(), int(r["label"]),
                         (r.get("language") or "").strip().lower()))
    return rows


def read_pool(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            body = rec["state"][len("Premise: "):]
            p, h = body.split("\n\nHypothesis: ", 1)
            lab = rec["questions"]["relation"]["label"]
            rows.append((p.strip(), h.strip(), lab if isinstance(lab, int) else C2I[lab]))
    return rows


comp_rows = read_comp_train()
log("competition train rows %d" % len(comp_rows))
train4 = [(p, h, y) for p, h, y, _ in comp_rows]
if EN_REPEAT:
    en = [(p, h, y) for p, h, y, lg in comp_rows if lg == "english"]
    train4 += en * EN_REPEAT
    log("en-repeat x%d: +%d English rows" % (EN_REPEAT, len(en) * EN_REPEAT))
if POOL:
    pool_rows = read_pool(POOL)
    log("pool rows %d" % len(pool_rows))
    train4 = train4 + pool_rows
train_rows = train4
log("total train rows %d  labels %s" % (len(train_rows),
      {c: sum(1 for r in train_rows if r[2] == i) for i, c in enumerate(CLASSES)}))

test_rows = []
with open(os.path.join(COMP_DIR, "test.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        test_rows.append((r["id"], (r["premise"] or "").strip(), (r["hypothesis"] or "").strip()))
log("test rows %d" % len(test_rows))


# ------------------------------------------------------------------ model
torch.manual_seed(SEED)
np.random.seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=len(CLASSES))
if GRAD_CKPT:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    log("gradient checkpointing ON")
if FREEZE_BELOW >= 0:
    import re as _re
    _frozen = 0
    for _nm, _p in model.named_parameters():
        if "embeddings" in _nm:
            _p.requires_grad = False
            _frozen += _p.numel()
            continue
        _m = _re.search(r"encoder\.layer\.(\d+)\.", _nm)
        if _m and int(_m.group(1)) < FREEZE_BELOW:
            _p.requires_grad = False
            _frozen += _p.numel()
    log("freeze-below %d: %.1fM params frozen, %.1fM trainable"
        % (FREEZE_BELOW, _frozen / 1e6,
           sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEV)
log("model params %.1fM on %s" % (sum(p.numel() for p in model.parameters()) / 1e6, DEV))

try:                                   # sm_75 (T4) has no bf16 tensor cores
    AMP_DTYPE = torch.bfloat16 if (DEV == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
except Exception:  # noqa: BLE001
    AMP_DTYPE = torch.float16
log("amp dtype %s" % AMP_DTYPE)


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
              padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
    enc["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return enc


@torch.no_grad()
def predict(pairs, bs=64):
    model.eval()
    out = []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        enc = tok([c[0] for c in chunk], [c[1] for c in chunk], padding=True, truncation=True,
                  max_length=MAXLEN, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEV == "cuda")):
            logits = model(**enc).logits
        out += torch.softmax(logits.float(), dim=-1).cpu().numpy().tolist()
    return out


dl = DataLoader(DS(train_rows), batch_size=BATCH, shuffle=True, collate_fn=collate)
decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (no_decay if (n.endswith("bias") or "LayerNorm" in n or "layer_norm" in n) else decay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                         {"params": no_decay, "weight_decay": 0.0}], lr=LR, betas=(0.9, 0.98), eps=1e-6)

updates_per_epoch = max(1, math.ceil(len(dl) / ACCUM))
total_steps = max(1, int(updates_per_epoch * EPOCHS))
warm = max(1, int(total_steps * 0.06))


def lr_at(step):
    if step < warm:
        return LR * (step + 1) / warm
    prog = (step - warm) / max(1, total_steps - warm)
    return LR * max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))


try:
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16 and DEV == "cuda"))
except AttributeError:
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16 and DEV == "cuda"))

log("training %d updates (%d epochs x %d rows)" % (total_steps, EPOCHS, len(train_rows)))
model.train()
step = 0
running, seen = 0.0, 0
opt.zero_grad(set_to_none=True)
t0 = time.time()
stop = False
for ep in range(int(math.ceil(EPOCHS))):
    for i, batch in enumerate(dl):
        batch = {k: v.to(DEV) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEV == "cuda")):
            loss = model(**batch).loss / ACCUM
        scaler.scale(loss).backward()
        running += loss.item() * ACCUM
        seen += 1
        if (i + 1) % ACCUM == 0 or (i + 1) == len(dl):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0 or step == total_steps:
                el = time.time() - t0
                log("step %d/%d loss %.4f lr %.2e %.2fs/upd elapsed %.1fmin"
                    % (step, total_steps, running / max(1, seen), lr_at(step), el / max(1, step), el / 60))
                running, seen = 0.0, 0
            if step >= total_steps:
                stop = True
                break
    if stop:
        break
log("trained %d updates in %.1f min" % (step, (time.time() - t0) / 60))

# ------------------------------------------------------------------ predict + write submission
probs = predict([(p, h) for _, p, h in test_rows])
preds = [int(np.argmax(p)) for p in probs]
with open(SUBMISSION, "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "prediction"])
    for (rid, _, _), y in zip(test_rows, preds):
        w.writerow([rid, y])
from collections import Counter
log("wrote %s  %d rows  dist %s" % (SUBMISSION, len(preds), dict(Counter(preds))))
