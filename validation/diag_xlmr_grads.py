"""Why does XLM-R large not learn? Measure, don't guess.

Runs a handful of steps and reports, per layer: grad L2 norm, and the actual parameter update
magnitude relative to the stored precision. If grads are healthy but weights don't move, the
optimizer/precision is at fault; if grads vanish, the graph is.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_hf_nli2 import load_rows  # noqa: E402

ROOT = Path("E:/Bill/AI/AutoKaggle")
MODEL = str(ROOT / "models/xlm-roberta-large")
DATA = str(ROOT / "data/comp/contradictory-my-dear-watson/kev_train_pool.jsonl")


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    lr = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-5
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    precision = sys.argv[4] if len(sys.argv) > 4 else "fp32"

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    print("model dtype:", next(model.parameters()).dtype, "device:", dev, flush=True)

    rows = load_rows(DATA, 400)
    enc = [tok(p, h, truncation=True, max_length=256) for p, h, _ in rows]
    labels = [r[2] for r in rows]

    def batch(i):
        ch = enc[i:i + bs]
        n = max(len(c["input_ids"]) for c in ch)
        ids = torch.zeros(len(ch), n, dtype=torch.long)
        am = torch.zeros(len(ch), n, dtype=torch.long)
        for r, c in enumerate(ch):
            ids[r, :len(c["input_ids"])] = torch.tensor(c["input_ids"])
            am[r, :len(c["input_ids"])] = 1
        return (ids.to(dev), am.to(dev),
                torch.tensor(labels[i:i + bs], dtype=torch.long).to(dev))

    optim_name = sys.argv[5] if len(sys.argv) > 5 else "adamw8bit"
    ps = [p for p in model.parameters() if p.requires_grad]
    if optim_name == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(ps, lr=lr, betas=(0.9, 0.98), eps=1e-8)
    else:
        opt = torch.optim.AdamW(ps, lr=lr, betas=(0.9, 0.98), eps=1e-8)
    watch = {
        "emb": "roberta.embeddings.word_embeddings.weight",
        "L00": "roberta.encoder.layer.0.attention.self.query.weight",
        "L12": "roberta.encoder.layer.12.attention.self.query.weight",
        "L23": "roberta.encoder.layer.23.attention.self.query.weight",
        "head": "classifier.out_proj.weight",
    }
    before = {k: model.get_parameter(v).detach().clone() for k, v in watch.items()}

    model.train()
    use_amp = precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for s in range(steps):
        ids, am, y = batch((s * bs) % (len(rows) - bs))
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=ids, attention_mask=am, labels=y)
            loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = {}
        for k, v in watch.items():
            g = model.get_parameter(v).grad
            gn[k] = float(g.norm()) if g is not None else -1.0
        with torch.no_grad():
            lg = out.logits.float()
            lstd = float(lg.std(0).mean())
        opt.step()
        if s % 5 == 0 or s == steps - 1:
            print("step %3d loss %.5f logit_std %.4f  gradnorm %s"
                  % (s, float(loss), lstd,
                     " ".join("%s=%.2e" % (k, gn[k]) for k in watch)), flush=True)

    print("\nweight movement after %d steps (abs max |dw| / |w|)" % steps)
    for k, v in watch.items():
        w0 = before[k]
        w1 = model.get_parameter(v).detach()
        d = (w1 - w0).abs()
        print("  %-4s |w|mean %.5f  |dw|max %.3e  |dw|mean %.3e  rel %.2e"
              % (k, float(w0.abs().mean()), float(d.max()), float(d.mean()),
                 float(d.mean() / max(1e-12, float(w0.abs().mean())))))


if __name__ == "__main__":
    main()
