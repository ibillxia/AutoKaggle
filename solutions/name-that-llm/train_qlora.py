"""Local 4B QLoRA sanity run for 18-way authorship classification.

Why local: Kaggle has no bitsandbytes, so the 9B route there has to be full fp16 across 2xT4.
This script answers the cheaper question first — does discriminative fine-tuning actually clear
the 0.44 plateau? — on this machine's 8 GB card with 4-bit weights.

Usage:
  python train_qlora.py --epochs 3 --out runs/ntl-qlora-4b
Loads the grouped holdout (30 unseen prompts) for an honest score after every epoch.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]


class Head(nn.Module):
    """Qwen body + mean-pooled classification head, kept separate so LoRA only touches the body."""

    def __init__(self, base, hidden, n_cls, dropout=0.1):
        super().__init__()
        self.base = base
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_cls)

    def forward(self, input_ids, attention_mask):
        hs = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.fc(self.drop(pooled.float()))


def batches(idx, bs, shuffle, rng):
    idx = np.array(idx)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="E:/Bill/AI/AutoKaggle/models/Qwen3.5-4B-Base")
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--full", type=int, default=0, help="1 = fit on all rows (final model)")
    ap.add_argument("--out", default="runs/ntl-qlora-4b")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    df = pd.read_csv(a.data)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, ho_idx = next(gss.split(df, df["model"], groups=df["prompt_id"]))
    if a.full:
        tr_idx = np.arange(len(df))
    train, hold = df.iloc[tr_idx].reset_index(drop=True), df.iloc[ho_idx].reset_index(drop=True)
    y_train = np.array([MODELS.index(m) for m in train["model"]])
    y_hold = np.array([MODELS.index(m) for m in hold["model"]])
    print(f"train {len(train)}  holdout {len(hold)}", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModel.from_pretrained(a.model, quantization_config=bnb, device_map={"": 0})
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    lcfg = LoraConfig(r=a.lora, lora_alpha=2 * a.lora, lora_dropout=0.05, bias="none",
                      target_modules="all-linear")
    base = get_peft_model(base, lcfg)
    base.print_trainable_parameters()

    model = Head(base, base.config.text_config.hidden_size, len(MODELS)).cuda()
    model.base.gradient_checkpointing_enable()
    params = [{"params": [p for n, p in model.named_parameters() if "fc" in n], "lr": a.head_lr},
              {"params": [p for n, p in model.named_parameters() if "fc" not in n and p.requires_grad],
               "lr": a.lr}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = (len(train) + a.bs - 1) // a.bs * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in params],
                                                total_steps=steps, pct_start=0.05)

    text_cache = train["text"].tolist()

    def encode(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=a.maxlen, return_tensors="pt")
        return enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    @torch.no_grad()
    def evaluate(texts, bs=8):
        model.eval()
        probs = []
        for i in range(0, len(texts), bs):
            ids, am = encode(texts[i:i + bs])
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(ids, am)
            probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
        model.train()
        return np.concatenate(probs)

    rng = np.random.default_rng(0)
    best = (-1, -1)
    for ep in range(a.epochs):
        t0, run = time.time(), 0.0
        model.train()
        for i, batch in enumerate(batches(np.arange(len(train)), a.bs, True, rng)):
            ids, am = encode([text_cache[j] for j in batch])
            y = torch.tensor(y_train[batch]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = nn.functional.cross_entropy(model(ids, am), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run += float(loss)
            if i % 20 == 0:
                el = time.time() - t0
                print(f"ep{ep} step {i}/{steps // a.epochs} loss {run / (i + 1):.4f} "
                      f"{el / (i + 1) * 1000:.0f} ms/step", flush=True)
        if len(hold):
            P = evaluate(hold["text"].tolist())
            acc = balanced_accuracy_score(hold["model"].values, [MODELS[i] for i in P.argmax(1)])
            print(f"== ep{ep} holdout bal_acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if acc > best[0]:
                best = (acc, ep)
                model.base.save_pretrained(out / "adapter")
                torch.save(model.fc.state_dict(), out / "head.pt")
                (out / "config.json").write_text(json.dumps(
                    {"models": MODELS, "lora": a.lora, "maxlen": a.maxlen, "epoch": ep}), encoding="utf-8")
        else:
            model.base.save_pretrained(out / "adapter")
            torch.save(model.fc.state_dict(), out / "head.pt")
            (out / "config.json").write_text(json.dumps(
                {"models": MODELS, "lora": a.lora, "maxlen": a.maxlen, "full": 1}), encoding="utf-8")
    print("best:", best, flush=True)


if __name__ == "__main__":
    main()
