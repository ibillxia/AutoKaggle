"""Can XLM-R large memorise 512 rows at all? If not, the pipeline is structurally broken, not
under-trained. Prints train accuracy + loss every N steps and the width of the pooled representation.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_hf_nli2 import load_rows  # noqa: E402

ROOT = Path("E:/Bill/AI/AutoKaggle")
MODEL = str(ROOT / "models/xlm-roberta-large")
DATA = str(ROOT / "data/comp/contradictory-my-dear-watson/kev_train_pool.jsonl")

n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 512
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 240
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-5
bs = int(sys.argv[4]) if len(sys.argv) > 4 else 8
freeze_below = int(sys.argv[5]) if len(sys.argv) > 5 else -1
lr_head = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-3

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
dev = "cuda"
model.to(dev)

if freeze_below >= 0:
    import re
    for nm, p in model.named_parameters():
        if "embeddings" in nm:
            p.requires_grad = False
        else:
            m = re.search(r"encoder\.layer\.(\d+)\.", nm)
            if m and int(m.group(1)) < freeze_below:
                p.requires_grad = False
    print("freeze_below", freeze_below, flush=True)

enc_groups = [{"params": [p for n, p in model.named_parameters()
                          if not n.startswith("classifier") and p.requires_grad], "lr": lr}]
head_groups = [{"params": [p for n, p in model.named_parameters()
                           if n.startswith("classifier")], "lr": lr_head}]
import bitsandbytes as bnb  # noqa: E402
opt = bnb.optim.AdamW8bit(enc_groups + head_groups, lr=lr, betas=(0.9, 0.98), eps=1e-8)

rows = load_rows(DATA, n_rows)
enc = [tok(p, h, truncation=True, max_length=256) for p, h, _ in rows]
labels = torch.tensor([r[2] for r in rows])


def batch(idx):
    ch = [enc[i] for i in idx]
    n = max(len(c["input_ids"]) for c in ch)
    ids = torch.zeros(len(ch), n, dtype=torch.long)
    am = torch.zeros(len(ch), n, dtype=torch.long)
    for r, c in enumerate(ch):
        ids[r, :len(c["input_ids"])] = torch.tensor(c["input_ids"])
        am[r, :len(c["input_ids"])] = 1
    return ids.to(dev), am.to(dev), labels[idx].to(dev)


perm_all = torch.randperm(len(rows))
model.train()
for s in range(steps):
    idx = perm_all[(s * bs) % (len(rows) - bs):(s * bs) % (len(rows) - bs) + bs]
    ids, am, y = batch(idx)
    out = model(input_ids=ids, attention_mask=am, labels=y)
    opt.zero_grad(set_to_none=True)
    out.loss.backward()
    opt.step()
    if s % 20 == 0 or s == steps - 1:
        with torch.no_grad():
            accs = []
            for i in range(0, 128, 64):
                ids, am, y = batch(perm_all[i:i + 64])
                o = model(input_ids=ids, attention_mask=am)
                lg = o.logits.float()
                accs.append((lg.argmax(-1) == y).float())
                last_std = float(lg.std(0).mean())
            acc = float(torch.cat(accs).mean())
            extra = ""
        print("step %3d loss %.4f logit_std %.3f  trainacc@128 %.3f%s"
              % (s, float(out.loss), last_std, acc, extra), flush=True)

# final full-set accuracy
model.eval()
correct = 0
with torch.no_grad():
    for i in range(0, len(rows), 64):
        ids, am, y = batch(list(range(i, min(i + 64, len(rows)))))
        correct += int((model(input_ids=ids, attention_mask=am).logits.argmax(-1) == y).sum())
print("FINAL train acc on all %d rows: %.4f" % (len(rows), correct / len(rows)))
