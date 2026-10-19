"""Smoke test: load a kev checkpoint and score one typed decision."""
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")      # bases are local now; never touch the network
sys.path.insert(0, "E:/Bill/AI/AutoKaggle/third_party/kev")

import torch
from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import materialize

RUN = sys.argv[1] if len(sys.argv) > 1 else "E:/Bill/AI/AutoKaggle/models/kev-0.8b-local"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

t0 = time.time()
ck = Checkpoint(RUN)
print(f"checkpoint  : {ck.path}")
print(f"base        : {ck.meta.base}")
print(f"lora rank   : {ck.meta.lora}   head_dim={ck.meta.head_dim}  T={ck.meta.temperature:.4f}")
print(f"hybrid base : {ck.hybrid_base()}")

opts = LoadOptions(dtype=torch.bfloat16 if DEV == "cuda" else None, backend="torch")
tok, model = ck.load(DEV, opts)
print(f"loaded in {time.time() - t0:.1f}s on {DEV}")
if DEV == "cuda":
    print(f"VRAM in use : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

req = {
    "state": "The shoes arrived two weeks late and in completely the wrong size. I want a refund.",
    "questions": {
        "dept": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                              "shipping": "Delivery status, delays, lost packages",
                              "billing": "Charges, invoices, payment problems"},
                 "label": "returns", "src": "smoke"},
        "urgent": {"type": "noul", "instructions": "Does this need urgent human attention?",
                   "label": True, "src": "smoke"},
        "angry": {"type": "score", "instructions": "How frustrated is the customer?",
                  "criteria": ["Calm", "Frustrated", "Very angry"], "label": 1, "src": "smoke"},
    },
}

template = materialize(req)
for q, (qid, src_q) in zip(template["questions"], req["questions"].items()):
    print(f"  q {qid}: qtype={q['qtype']} label_idx={q['label']} keys={q['keys']}")

enc = model.encode(tok, template, strict=True)
print(f"state tokens={enc['state_tokens']}  packed={len(enc['ids'])}  branches={len(enc['decide_idx'])}")
t1 = time.time()
probs = model.probs(enc)
print(f"scored {len(probs)} questions in {time.time() - t1:.3f}s")

for (qid, src_q), q, p in zip(req["questions"].items(), template["questions"], probs):
    p = p.float().cpu().numpy()
    y = q["label"]
    print(f"  {qid:8s} keys={q['keys']}")
    print(f"           probs={[round(float(x), 4) for x in p]}  argmax={q['keys'][int(p.argmax())]}  label={q['keys'][y] if q['qtype']=='choice' else y}  p(label)={p[y]:.4f}")
print("SMOKE OK")
