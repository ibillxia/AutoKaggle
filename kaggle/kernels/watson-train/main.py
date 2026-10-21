"""Kaggle GPU kernel: train XLM-R large on the watson NLI corpus, then score test.csv.

Designed for a single 2 h session: the training loop is time-budgeted, so it always leaves
room to predict and write submission.csv. Everything runs offline -- the base checkpoint and the
corpus both come from datasets, the test set comes from the competition mount.
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
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASSES = ["entailment", "neutral", "contradiction"]
L2I = {c: i for i, c in enumerate(CLASSES)}

DATA_FILE = "kev_train_huge.jsonl"
FREEZE_BELOW = 0
EPOCHS = 0.55
LR = 2e-05
HEAD_LR = 0.0001
LAYER_DECAY = 0.8
BATCH = 20
ACCUM = 1
MAXLEN = 192
WD = 0.01
CLIP = 1.0
WARMUP_FRAC = 0.05
TIME_BUDGET_MIN = 90
SEED = 0
SAVE_MODEL = 1

WORK = "/kaggle/working"
t0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - t0, msg), flush=True)


def find_competition_dir():
    for sub in glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True):
        d = os.path.dirname(sub)
        if os.path.isfile(os.path.join(d, "test.csv")):
            return d
    raise SystemExit("competition data not found")


def find_base_model():
    for c in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(c)
        try:
            meta = json.load(open(c, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if meta.get("model_type") != "xlm-roberta":
            continue
        if os.path.isfile(os.path.join(d, "model.safetensors")) or \
                os.path.isfile(os.path.join(d, "pytorch_model.bin")):
            if meta.get("num_hidden_layers") == 24:
                return d
    raise SystemExit("xlm-roberta-large not found")


COMP = find_competition_dir()
BASE = find_base_model()
data_path = ""
for c in glob.glob("/kaggle/input/**/" + DATA_FILE, recursive=True):
    data_path = c
    break
if not data_path:
    raise SystemExit("training file %s not found" % DATA_FILE)
log("competition %s" % COMP)
log("base model  %s" % BASE)
log("data        %s" % data_path)
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_rows(path):
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
            y = r["questions"]["relation"]["label"]
            prem.append(p)
            hyp.append(h)
            lab.append(L2I[y])
    return prem, hyp, np.asarray(lab, dtype=np.int64)


tok = AutoTokenizer.from_pretrained(BASE)
prem, hyp, labels = load_rows(data_path)
log("rows %d" % len(labels))

ids_buf, mask_buf = [], []
B = 2048
for i in range(0, len(prem), B):
    enc = tok(prem[i:i + B], hyp[i:i + B], truncation=True, max_length=MAXLEN,
              padding="max_length", return_tensors="pt")
    ids_buf.append(enc["input_ids"].to(torch.int32))
    mask_buf.append(enc["attention_mask"].to(torch.int16))
    if (i // B) % 40 == 0:
        log("tokenised %d/%d" % (min(i + B, len(prem)), len(prem)))
IDS = torch.cat(ids_buf)
MASK = torch.cat(mask_buf)
del ids_buf, mask_buf, prem, hyp
log("tokenised %d rows  shape %s" % (IDS.shape[0], tuple(IDS.shape[1:])))

model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(CLASSES))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEV)

# freeze the requested number of encoder layers (0 = full fine-tuning)
if FREEZE_BELOW > 0:
    enc = model.roberta
    for p in enc.embeddings.parameters():
        p.requires_grad = False
    for layer in enc.encoder.layer[:FREEZE_BELOW]:
        for p in layer.parameters():
            p.requires_grad = False
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
log("trainable %.1fM / %.1fM" % (n_train / 1e6,
                                 sum(p.numel() for p in model.parameters()) / 1e6))

groups = {}
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    depth = 24
    if ".layer." in name:
        try:
            depth = int(name.split(".layer.")[1].split(".")[0])
        except ValueError:
            depth = 0
    elif "embeddings" in name:
        depth = 0
    if name.startswith("classifier") or "pooler" in name:
        groups.setdefault("head", []).append(p)
    else:
        scale = LAYER_DECAY ** (23 - depth) if LAYER_DECAY < 1.0 else 1.0
        key = "%.6f" % (LR * scale)
        groups.setdefault(key, []).append(p)
param_groups = [{"params": v, "lr": HEAD_LR if k == "head" else float(k)}
                for k, v in groups.items()]
log("param groups %d" % len(param_groups))

try:
    import bitsandbytes as bnb  # noqa: F401
    opt = bnb.optim.AdamW8bit(param_groups, lr=LR, weight_decay=WD)
    log("optimiser AdamW8bit")
except Exception as exc:  # noqa: BLE001
    log("bitsandbytes unavailable (%s) -> torch AdamW" % type(exc).__name__)
    opt = torch.optim.AdamW(param_groups, lr=LR, weight_decay=WD)
scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))

n = len(labels)
steps_per_epoch = math.ceil(n / (BATCH * ACCUM))
est_total = steps_per_epoch * EPOCHS
budget_s = TIME_BUDGET_MIN * 60
log("planned total steps %d (%.1f epoch(s))" % (est_total, EPOCHS))


def lr_scale(step):
    if step < est_total * WARMUP_FRAC:
        return (step + 1) / max(1.0, est_total * WARMUP_FRAC)
    prog = (step - est_total * WARMUP_FRAC) / max(1.0, est_total * (1 - WARMUP_FRAC))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


base_lrs = [g["lr"] for g in opt.param_groups]
model.train()
rng = np.random.RandomState(SEED)
step = 0
micro = 0
run_loss, run_n = 0.0, 0
t_start = time.time()
stop = False
ep = 0
while not stop and ep < math.ceil(EPOCHS):
    perm = rng.permutation(n)
    for s in range(0, n, BATCH):
        if step >= est_total:
            stop = True
            break
        if time.time() - t_start > budget_s:
            log("time budget hit at step %d" % step)
            stop = True
            break
        idx = perm[s:s + BATCH]
        inp = IDS[idx].to(DEV, non_blocking=True).long()
        am = MASK[idx].to(DEV, non_blocking=True).long()
        y = torch.from_numpy(labels[idx]).to(DEV)
        sc = lr_scale(step)
        for g, blr in zip(opt.param_groups, base_lrs):
            g["lr"] = blr * sc
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
            out = model(input_ids=inp, attention_mask=am, labels=y)
            loss = out.loss / ACCUM
        scaler.scale(loss).backward()
        run_loss += float(loss) * ACCUM
        run_n += 1
        micro += 1
        if micro % ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CLIP)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                el = time.time() - t_start
                log("step %d/%d  loss %.4f  lr %.2e  %.3fs/step  %.1f min"
                    % (step, est_total, run_loss / max(1, run_n), opt.param_groups[0]["lr"],
                       el / step, el / 60))
                run_loss, run_n = 0.0, 0
    ep += 1
log("trained %d steps in %.1f min" % (step, (time.time() - t_start) / 60))

model.eval()
test_rows = []
with open(os.path.join(COMP, "test.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        test_rows.append((r["id"], (r.get("premise") or "").strip(),
                          (r.get("hypothesis") or "").strip()))
probs = []
with torch.no_grad():
    for i in range(0, len(test_rows), 64):
        chunk = test_rows[i:i + 64]
        enc = tok([c[1] for c in chunk], [c[2] for c in chunk], padding=True, truncation=True,
                  max_length=MAXLEN, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
            lg = model(**enc).logits
        probs += torch.softmax(lg.float(), -1).cpu().numpy().tolist()
preds = [int(np.argmax(p)) for p in probs]
with open(os.path.join(WORK, "submission.csv"), "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "prediction"])
    for (rid, _, _), y in zip(test_rows, preds):
        w.writerow([rid, y])
log("wrote submission.csv %d rows dist %s" % (len(preds), dict(Counter(preds))))
with open(os.path.join(WORK, "test_probs.jsonl"), "w", encoding="utf-8") as fh:
    for (rid, _, _), p in zip(test_rows, probs):
        fh.write(json.dumps({"id": rid, "p": p}) + "\n")

if SAVE_MODEL:
    model.save_pretrained(os.path.join(WORK, "model"))
    tok.save_pretrained(os.path.join(WORK, "model"))
    log("saved model to %s/model" % WORK)
