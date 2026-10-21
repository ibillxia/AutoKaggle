# Fine-tune a bidirectional encoder (roberta-large / ModernBERT) for 18-way attribution.
# Same grouped holdout as prep_kev.py so numbers are comparable with TF-IDF and kev.
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="enc")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--full", type=int, default=0, help="1 = train on every row (final model, no holdout)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=len(MODELS),
        id2label={i: m for i, m in enumerate(MODELS)},
        label2id={m: i for i, m in enumerate(MODELS)}).to(device)

    df = pd.read_csv("data/train.csv")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, ho_idx = next(gss.split(df, df["model"], groups=df["prompt_id"]))
    if a.full:
        tr_idx, ho_idx = np.arange(len(df)), np.arange(min(64, len(df)))
    tr, ho = df.iloc[tr_idx].reset_index(drop=True), df.iloc[ho_idx].reset_index(drop=True)

    def encode(texts):
        return tok(list(texts), padding="max_length", truncation=True, max_length=a.maxlen,
                   return_tensors="pt")

    Xtr, Xho = encode(tr["text"]), encode(ho["text"])
    ytr = torch.tensor([MODELS.index(m) for m in tr["model"]])
    yho = torch.tensor([MODELS.index(m) for m in ho["model"]])

    ds = torch.utils.data.TensorDataset(Xtr["input_ids"], Xtr["attention_mask"], ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.06)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(a.dtype)
    use_amp = amp_dtype is not None and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(a.dtype == "fp16"))

    for ep in range(a.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for i, (ids, am, y) in enumerate(dl):
            ids, am, y = ids.to(device), am.to(device), y.to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = model(input_ids=ids, attention_mask=am, labels=y).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            run += loss.item()
            if (i + 1) % 20 == 0:
                print(f"  ep{ep} {i+1}/{len(dl)} loss={run/(i+1):.4f} {(time.time()-t0)/(i+1):.2f}s/it", flush=True)
        # holdout check each epoch
        model.eval()
        preds, probs = [], []
        with torch.no_grad():
            for i in range(0, len(ho), 32):
                b = {k: v[i:i + 32].to(device) for k, v in Xho.items()}
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(**b).logits
                p = torch.softmax(logits.float(), -1).cpu()
                probs.append(p)
                preds += p.argmax(1).tolist()
        p = torch.cat(probs).numpy()
        acc = balanced_accuracy_score(yho.numpy(), np.array(preds))
        print(f"[ep{ep}] holdout bal_acc={acc:.4f} ({(time.time()-t0)/60:.1f} min this epoch)", flush=True)
        if a.out:
            model.save_pretrained(a.out)
            tok.save_pretrained(a.out)
            with open(a.out + "/holdout_probs.npz", "wb") as f:
                np.savez(f, probs=p, y=yho.numpy(), classes=np.array(MODELS))
            print(f"  saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
