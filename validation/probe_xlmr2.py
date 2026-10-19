"""Why does joint fine-tuning of XLM-R large not move, while a linear probe on its frozen features
reaches 100% on the same 128 rows?

Measures, on a *fixed* batch (no shuffling, so the loss trajectory is clean):
  * how many parameters actually receive a gradient
  * per-parameter gradient norm for a head weight vs. encoder weights at each depth
  * how far each weight drifts after N Adam steps
  * the same run under autocast-bf16 vs pure fp32

If the encoder grads are zero / drifts are zero -> the encoder is detached from the loss.
If drift is normal but the fixed-batch loss still won't fall -> the updates are destroying the
features faster than the head can exploit them (i.e. the LR is far too high for this setup).
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "xlm-roberta-large"
DATA = ROOT / "data/comp/contradictory-my-dear-watson/kev_train_pool.jsonl"
CLASSES = ["entailment", "neutral", "contradiction"]

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--max-len", type=int, default=64)
ap.add_argument("--freeze-embeddings", type=int, default=1)
ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rows = []
for line in DATA.open(encoding="utf-8"):
    rec = json.loads(line)
    body = rec["state"][len("Premise: "):]
    p, h = body.split("\n\nHypothesis: ", 1)
    lab = rec["questions"]["relation"]["label"]
    rows.append((p.strip(), h.strip(), CLASSES.index(lab) if isinstance(lab, str) else int(lab)))
    if len(rows) >= a.batch:
        break

torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
enc = tok([r[0] for r in rows], [r[1] for r in rows], padding=True, truncation=True,
          max_length=a.max_len, return_tensors="pt").to("cuda")
ys = torch.tensor([r[2] for r in rows], device="cuda")
enc["labels"] = ys

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
if a.freeze_embeddings:
    for n, p in model.named_parameters():
        if "embeddings" in n:
            p.requires_grad = False
model.to("cuda")
model.train()

allp = [p for p in model.parameters() if p.requires_grad]
n_all = sum(p.numel() for p in model.parameters())
n_tr = sum(p.numel() for p in allp)
print("trainable %.1fM / %.1fM  precision=%s" % (n_tr / 1e6, n_all / 1e6, a.precision), flush=True)

watch = {}
for n, p in model.named_parameters():
    if n.endswith("classifier.out_proj.weight"):
        watch["head.out_proj"] = n
    if n.endswith("classifier.dense.weight"):
        watch["head.dense"] = n
    if "encoder.layer.0.attention.self.query.weight" in n:
        watch["enc.L0.q"] = n
    if "encoder.layer.12.attention.self.query.weight" in n:
        watch["enc.L12.q"] = n
    if "encoder.layer.23.output.dense.weight" in n:
        watch["enc.L23.o"] = n
named = dict(model.named_parameters())
before = {k: named[v].detach().clone().float() for k, v in watch.items()}

opt = torch.optim.AdamW(allp, lr=a.lr, betas=(0.9, 0.98), eps=1e-6)


def step_loss():
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.precision == "bf16")):
        out = model(**enc)
    return out


for step in range(a.steps + 1):
    out = step_loss()
    loss = out.loss
    if step == a.steps:
        break
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if step in (0, 4, a.steps - 1):
        ngrad = sum(1 for p in model.parameters() if p.grad is not None)
        tot = sum(1 for _ in model.parameters())
        gs = {k: (named[v].grad.norm().item() if named[v].grad is not None else None)
              for k, v in watch.items()}
        mr = max((p.grad.norm().item() for p in allp if p.grad is not None), default=0.0)
        raw = torch.nn.utils.clip_grad_norm_(allp, 1.0).item()
        print("  step %2d loss %.4f acc %.3f | grads on %d/%d tensors | raw_gnorm %.2f max_tensor_grad %.2f"
              % (step, loss.item(), (out.logits.argmax(-1) == ys).float().mean().item(),
                 ngrad, tot, raw, mr), flush=True)
        print("        %s" % {k: (round(v, 5) if v is not None else None) for k, v in gs.items()}, flush=True)
    else:
        torch.nn.utils.clip_grad_norm_(allp, 1.0)
    opt.step()

print("--- weight drift after %d steps (L2 over the tensor) ---" % a.steps, flush=True)
for k, v in watch.items():
    d = (named[v].detach().float() - before[k]).norm().item()
    s = before[k].norm().item()
    print("  %-12s drift %.6f  (weight norm %.4f, rel %.4f%%)" % (k, d, s, 100 * d / max(s, 1e-9)), flush=True)

out = step_loss()
print("FINAL fixed-batch loss %.4f  acc %.3f"
      % (out.loss.item(), (out.logits.argmax(-1) == ys).float().mean().item()), flush=True)
