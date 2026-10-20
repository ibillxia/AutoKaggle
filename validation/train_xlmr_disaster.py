"""XLM-R large on nlp-getting-started (binary "does this tweet report a disaster").

Why: the kev-0.8B/probability-ensemble route plateaued at LB 0.83113 (1->2->3 members:
0.82623 / 0.82745 / 0.83113). On contradictory-my-dear-watson the same conclusion held --
XLM-R large with the bottom 12 layers frozen beat every kev member (dev 0.8282 vs 0.8244)
and took the real LB to 0.82309. Tweets are short, so the whole run costs minutes, not hours.

Recipe copied from the watson winner (`validation/train_hf_nli.py`, fb12-pool config):
full fine-tune with `encoder.layer[:12]` + embeddings frozen, bf16, adamw8bit, warmup 6%,
batch 16. Full-depth full fine-tuning collapses on this card (logit_std ~0.14, loss ~ln 3),
and LoRA does not move on real data -- both were measured, see .workbuddy/memory/2026-10-19.md.

    python train_xlmr_disaster.py --epochs 3 --holdout 800
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COMP = ROOT / "data" / "comp" / "nlp-getting-started"


def parse(rec):
    q = rec["questions"]["disaster"]
    label = q["label"]
    return rec["state"].strip(), 1 if label else 0


def load(path):
    with Path(path).open(encoding="utf-8") as f:
        return [parse(json.loads(l)) for l in f if l.strip()]


def split(rows, holdout, seed):
    """Stratified holdout so the F1 read-out is not dominated by one class."""
    rng = random.Random(seed)
    by_cls = {0: [], 1: []}
    for i, (_, y) in enumerate(rows):
        by_cls[y].append(i)
    held = set()
    for cls, idx in by_cls.items():
        rng.shuffle(idx)
        k = int(round(holdout * len(idx) / len(rows)))
        held.update(idx[:k])
    tr = [r for i, r in enumerate(rows) if i not in held]
    ev = [r for i, r in enumerate(rows) if i in held]
    return tr, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/xlm-roberta-large")
    ap.add_argument("--data", default=str(COMP / "kev_train.jsonl"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--freeze-below", type=int, default=12)
    ap.add_argument("--holdout", type=int, default=800)
    ap.add_argument("--split-seed", type=int, default=7,
                    help="fixed so that several members share one holdout and can be averaged")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "runs" / "xlmr-disaster"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(a.seed)
    dev = a.device if torch.cuda.is_available() else "cpu"

    rows = load(a.data)
    if a.limit:
        rows = rows[: a.limit]
    tr, ev = split(rows, a.holdout if not a.limit else max(1, len(rows) // 5), a.split_seed)
    print(f"rows train={len(tr)} holdout={len(ev)}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(ROOT / a.model), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ROOT / a.model), num_labels=2, torch_dtype=torch.bfloat16)
    # the collapse guard: keep the bottom of the encoder frozen, train the top + classifier
    for p in model.roberta.embeddings.parameters():
        p.requires_grad = False
    for layer in model.roberta.encoder.layer[: a.freeze_below]:
        for p in layer.parameters():
            p.requires_grad = False
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"froze {frozen} tensors (below layer {a.freeze_below})", flush=True)
    model.to(dev)

    class DS(Dataset):
        def __init__(self, rs):
            self.rs = rs

        def __len__(self):
            return len(self.rs)

        def __getitem__(self, i):
            t, y = self.rs[i]
            return t, y

    def collate(batch):
        texts = [b[0] for b in batch]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = tok(texts, truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
        return enc, ys

    dl = DataLoader(DS(tr), batch_size=a.batch, shuffle=True, collate_fn=collate, drop_last=False)
    steps = max(1, int(math.ceil(len(dl) * a.epochs)))
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad],
                                  lr=a.lr, weight_decay=0.01)
        print("optim=adamw8bit", flush=True)
    except Exception:  # noqa: BLE001
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=a.lr, weight_decay=0.01)
        print("optim=adamw", flush=True)
    warm = max(1, int(0.06 * steps))

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        return a.lr * max(0.0, (steps - s) / max(1, steps - warm))

    def evaluate(rs, bs=64):
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for i in range(0, len(rs), bs):
                chunk = rs[i:i + bs]
                enc = tok([c[0] for c in chunk], truncation=True, max_length=a.max_len,
                          padding=True, return_tensors="pt").to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    logits = model(**enc).logits.float()
                ps.extend(torch.softmax(logits, -1)[:, 1].cpu().tolist())
                ys.extend([c[1] for c in chunk])
        from sklearn.metrics import f1_score
        pred = [1 if p >= 0.5 else 0 for p in ps]
        return f1_score(ys, pred), sum(int(p == y) for p, y in zip(pred, ys)) / len(ys), ps, ys

    step = 0
    t0 = time.time()
    model.train()
    for ep in range(int(math.ceil(a.epochs))):
        for enc, ys in dl:
            if step >= steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            enc = {k: v.to(dev) for k, v in enc.items()}
            ys = ys.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                loss = model(**enc, labels=ys).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"  step {step}/{steps} loss {loss.item():.4f} "
                      f"lr {lr_at(step):.2e} {(time.time() - t0) / step:.2f}s/step", flush=True)
        f1, acc, _, _ = evaluate(ev)
        print(f"[ep{ep + 1}] holdout F1={f1:.4f} acc={acc:.4f}", flush=True)

    f1, acc, ps, ys = evaluate(ev)
    print(f"FINAL holdout F1={f1:.4f} acc={acc:.4f}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    metrics = {"holdout_f1": f1, "holdout_acc": acc, "n_train": len(tr), "n_holdout": len(ev),
               "epochs": a.epochs, "lr": a.lr, "freeze_below": a.freeze_below, "max_len": a.max_len}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "holdout.jsonl").write_text(
        "".join(json.dumps({"label": y, "p": {"true": p}}) + "\n" for p, y in zip(ps, ys)),
        encoding="utf-8")
    print("saved", out, flush=True)

    # ---- test predictions, in test_ids.json order
    test_rows = [json.loads(l) for l in (COMP / "kev_test.jsonl").open(encoding="utf-8") if l.strip()]
    ids = json.loads((COMP / "test_ids.json").read_text(encoding="utf-8"))
    assert len(ids) == len(test_rows), (len(ids), len(test_rows))
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(test_rows), 64):
            chunk = test_rows[i:i + 64]
            enc = tok([c["state"] for c in chunk], truncation=True, max_length=a.max_len,
                      padding=True, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logits = model(**enc).logits.float()
            probs.extend(torch.softmax(logits, -1)[:, 1].cpu().tolist())
    target = ROOT / "submissions" / (Path(a.out).name + ".probs.jsonl")
    with target.open("w", encoding="utf-8") as f:
        for i, p in zip(ids, probs):
            f.write(json.dumps({"id": str(i), "pred": 1 if p >= 0.5 else 0,
                                "p": {"true": float(p)}}) + "\n")
    pos = sum(1 for p in probs if p >= 0.5)
    print(f"wrote {target}: {len(probs)} rows, positives={pos} ({pos / len(probs):.2%})", flush=True)


if __name__ == "__main__":
    main()
