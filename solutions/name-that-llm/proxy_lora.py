# Proxy experiment: does the forum corpus actually help a decoder-LM + pooled head?
# Same shape as the real recipe (stage-1 on external pool, stage-2 on official train),
# just on Qwen3.5-0.8B instead of 8B/14B so it fits an 8GB card.
#
#   python proxy_lora.py --stage1 pool --tag pool
#   python proxy_lora.py --stage1 pool_marg --tag poolmarg
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "E:/Bill/AI/AutoKaggle/models/Qwen3.5-0.8B-Base"
MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
L2I = {m: i for i, m in enumerate(MODELS)}


class DS(Dataset):
    def __init__(self, texts, labels, tok, maxlen):
        self.t = texts
        self.y = labels
        self.tok = tok
        self.maxlen = maxlen

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        enc = self.tok(self.t[i], truncation=True, max_length=self.maxlen,
                       return_tensors=None, add_special_tokens=True)
        return enc["input_ids"], self.y[i]


def collate(batch, pad_id):
    mx = max(len(b[0]) for b in batch)
    ids = torch.full((len(batch), mx), pad_id, dtype=torch.long)
    am = torch.zeros((len(batch), mx), dtype=torch.long)
    for i, (seq, _) in enumerate(batch):
        ids[i, :len(seq)] = torch.tensor(seq)
        am[i, :len(seq)] = 1
    y = torch.tensor([b[1] for b in batch])
    return ids, am, y


class Head(nn.Module):
    """Mean-pool over the last k hidden layers at the final token, then a linear map."""

    def __init__(self, hidden, n, k=8):
        super().__init__()
        self.k = k
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden, n)

    def forward(self, hs, am):
        last = am.sum(1) - 1
        vecs = []
        for h in hs[-self.k:]:
            vecs.append(h[torch.arange(h.size(0)), last])
        v = torch.stack(vecs).mean(0)
        return self.fc(self.drop(v.float()))


def build_stage1(kind, train, hold):
    uni = pd.read_parquet("extra/union_all.parquet")
    pool = uni[uni.in_pool_v1]
    marg = uni[(~uni.in_train) & (~uni.in_pool_v1)]
    bad = set(hold["text"].str.lower().str.strip()) | set(train["text"].str.lower().str.strip())
    pool = pool[~pool["text"].str.lower().str.strip().isin(bad)]
    marg = marg[~marg["text"].str.lower().str.strip().isin(bad)]
    if kind == "pool":
        return pool
    if kind == "marg":
        return marg
    if kind == "pool_marg":
        return pd.concat([pool, marg], ignore_index=True)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="pool", choices=["none", "pool", "marg", "pool_marg"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--ep1", type=int, default=1)
    ap.add_argument("--ep2", type=int, default=2)
    ap.add_argument("--lr1", type=float, default=1e-4)
    ap.add_argument("--lr2", type=float, default=5e-5)
    ap.add_argument("--cap", type=int, default=40000, help="cap stage-1 rows, class-balanced")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit2", type=int, default=0, help="smoke test: cap stage-2 rows")
    a = ap.parse_args()
    tag = a.tag or a.stage1

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    tr = pd.read_csv("data/train.csv")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    i1, i2 = next(gss.split(tr, tr["model"], groups=tr["prompt_id"]))
    train, hold = tr.iloc[i1].reset_index(drop=True), tr.iloc[i2].reset_index(drop=True)
    if a.limit2:
        train = train.sample(min(a.limit2, len(train)), random_state=0).reset_index(drop=True)
    print(f"train {len(train)} | holdout {len(hold)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    hidden = model.config.hidden_size

    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules="all-linear", task_type="FEATURE_EXTRACTION")
    model = get_peft_model(model, lcfg)
    head = Head(hidden, len(MODELS)).cuda()

    def run_stage(df, epochs, lr, bs, name):
        ds = DS(df["text"].tolist(), [L2I[m] for m in df["model"]], tok, a.maxlen)
        dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0,
                        collate_fn=lambda b: collate(b, tok.pad_token_id or tok.eos_token_id))
        params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        steps = max(1, epochs * len(dl))
        from transformers import get_cosine_schedule_with_warmup
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(1, steps // 20),
                                                num_training_steps=steps)
        model.train(); head.train()
        t0 = time.time()
        for ep in range(epochs):
            tot, n = 0.0, 0
            for ids, am, y in dl:
                ids, am, y = ids.cuda(), am.cuda(), y.cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=ids, attention_mask=am, output_hidden_states=True)
                    logits = head(out.hidden_states, am)
                loss = F.cross_entropy(logits, y, label_smoothing=0.05)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                tot += loss.item() * len(y); n += len(y)
            print(f"[{name}] ep{ep+1}/{epochs} loss={tot/n:.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{time.time()-t0:.0f}s", flush=True)

    if a.stage1 != "none":
        s1 = build_stage1(a.stage1, train, hold)
        # class-balanced cap so no model dominates stage 1
        per = max(1, a.cap // len(MODELS))
        s1 = s1.groupby("model", group_keys=False).apply(
            lambda g: g.sample(min(len(g), per), random_state=a.seed)).reset_index(drop=True)
        print(f"stage1[{a.stage1}] rows={len(s1)}", flush=True)
        run_stage(s1, a.ep1, a.lr1, a.bs, "s1")

    run_stage(train, a.ep2, a.lr2, a.bs, "s2")

    # eval
    model.eval(); head.eval()
    ds = DS(hold["text"].tolist(), [L2I[m] for m in hold["model"]], tok, a.maxlen)
    dl = DataLoader(ds, batch_size=32, shuffle=False,
                    collate_fn=lambda b: collate(b, tok.pad_token_id or tok.eos_token_id))
    preds = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ids, am, y in dl:
            out = model(input_ids=ids.cuda(), attention_mask=am.cuda(), output_hidden_states=True)
            preds.append(head(out.hidden_states, am.cuda()).float().argmax(-1).cpu().numpy())
    p = np.concatenate(preds)
    y = np.array([L2I[m] for m in hold["model"]])
    res = dict(tag=tag, stage1=a.stage1, rows=len(train), bal_acc=float(balanced_accuracy_score(y, p)),
               macro_f1=float(f1_score(y, p, average="macro")))
    print("RESULT " + json.dumps(res), flush=True)
    os.makedirs("extra/proxy", exist_ok=True)
    with open(f"extra/proxy/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    np.save(f"extra/proxy/{tag}_pred.npy", p)


if __name__ == "__main__":
    main()
