"""Is the XLM-R fine-tune failing because the tokenizer is degenerate, or because the pretrained
features are garbage, or purely an optimisation problem?

Three probes, cheapest first:
  1. tokenizer sanity -- do different inputs produce different, non-trivial id sequences?
  2. frozen-encoder linear probe -- can a single linear layer separate 128 rows of real features?
     If yes, the backbone is fine and only joint fine-tuning is broken.
  3. feature statistics -- mean/std of the pooled sentence vector.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "xlm-roberta-large"
DATA = ROOT / "data/comp/contradictory-my-dear-watson/kev_train_pool.jsonl"
CLASSES = ["entailment", "neutral", "contradiction"]

rows = []
for line in DATA.open(encoding="utf-8"):
    rec = json.loads(line)
    body = rec["state"][len("Premise: "):]
    p, h = body.split("\n\nHypothesis: ", 1)
    lab = rec["questions"]["relation"]["label"]
    rows.append((p.strip(), h.strip(), CLASSES.index(lab) if isinstance(lab, str) else int(lab)))
    if len(rows) >= 128:
        break
print("rows", len(rows), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
print("tokenizer", type(tok).__name__, "len", len(tok), flush=True)

# ---- probe 1: tokenizer sanity -------------------------------------------------
probe = [
    ("The cat sat on the mat.", "A cat was on the mat."),
    ("The stock market crashed today.", "Shares fell sharply."),
    ("هذه الجملة باللغة العربية.", "هذه جملة مختلفة تماما."),
    ("今天天气很好。", "外面下着大雨。"),
]
for p, h in probe:
    enc = tok(p, h, truncation=True, max_length=256)
    ids = enc["input_ids"]
    dec = tok.decode(ids)
    print("  len=%3d unk=%d first16=%s" % (len(ids), sum(1 for i in ids if i == tok.unk_token_id), ids[:16]), flush=True)
    print("     decode: %s" % dec[:110], flush=True)

encs = [tok(p, h, truncation=True, max_length=256)["input_ids"] for p, h in probe]
print("  all identical?", all(e == encs[0] for e in encs), flush=True)

# ---- load model, freeze encoder ----------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3).to("cuda")
for n, p in model.named_parameters():
    if not n.startswith("classifier"):
        p.requires_grad = False
trainable = [p for p in model.parameters() if p.requires_grad]
print("trainable params %.2fM" % (sum(p.numel() for p in trainable) / 1e6), flush=True)

# ---- probe 2/3: frozen features -------------------------------------------------
xs = list(range(0, 128, 4))
enc = tok([rows[i][0] for i in xs], [rows[i][1] for i in xs], padding=True, truncation=True,
          max_length=128, return_tensors="pt").to("cuda")
ys = torch.tensor([rows[i][2] for i in xs], device="cuda")
with torch.no_grad():
    out = model.roberta(**enc)
feat = out.last_hidden_state[:, 0, :]
print("CLS feature std %.4f  mean-abs %.4f" % (feat.std().item(), feat.abs().mean().item()), flush=True)
print("head logits before: std %.4f" % model.classifier(out.last_hidden_state).std().item(), flush=True)

# ---- probe 2: linear probe on frozen features ---------------------------------
opt = torch.optim.AdamW(trainable, lr=1e-3)
model.train()
for step in range(201):
    logits = model.classifier(out.last_hidden_state)
    loss = torch.nn.functional.cross_entropy(logits, ys)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    if step % 40 == 0:
        acc = (logits.argmax(-1) == ys).float().mean().item()
        print("  LINEAR-PROBE step %3d  loss %.4f  acc %.3f  gnorm %.2f" % (step, loss.item(), acc, gn), flush=True)
print("LINEAR-PROBE FINAL loss %.4f acc %.3f"
      % (loss.item(), (logits.argmax(-1) == ys).float().mean().item()), flush=True)

# ---- probe 3: full joint fine-tune on the same 128 rows, high LR --------------
del model
torch.cuda.empty_cache()
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3).to("cuda")
model.train()
allp = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(allp, lr=5e-4)
for step in range(201):
    enc2 = tok([r[0] for r in rows[:32]], [r[1] for r in rows[:32]], padding=True, truncation=True,
               max_length=128, return_tensors="pt").to("cuda")
    y2 = torch.tensor([r[2] for r in rows[:32]], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out2 = model(**enc2, labels=y2)
    loss = out2.loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(allp, 1.0)
    opt.step()
    if step % 40 == 0:
        acc = (out2.logits.argmax(-1) == y2).float().mean().item()
        print("  JOINT lr=5e-4 step %3d  loss %.4f  acc %.3f  gnorm %.1f" % (step, loss.item(), acc, gn), flush=True)
print("JOINT FINAL loss %.4f acc %.3f"
      % (loss.item(), (out2.logits.argmax(-1) == y2).float().mean().item()), flush=True)
