# Encoder version of the data-value proxy: full fine-tune of ModernBERT-base with a
# pooled linear head, stage-1 on an external corpus then stage-2 on the official train.
# Same grouped holdout as screen_extra_tfidf.py so the numbers line up.
#
#   python proxy_enc.py --stage1 none        --tag enc_none
#   python proxy_enc.py --stage1 pool        --tag enc_pool
#   python proxy_enc.py --stage1 marg        --tag enc_marg
#   python proxy_enc.py --stage1 pool_marg   --tag enc_poolmarg
#   python proxy_enc.py --stage1 marg --only oleg_corpus --tag enc_marg_oleg
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
from transformers import AutoModel, AutoTokenizer

BASE = "E:/Bill/AI/AutoKaggle/models/ModernBERT-base"
MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
L2I = {m: i for i, m in enumerate(MODELS)}


class DS(Dataset):
    def __init__(self, texts, labels, tok, maxlen):
        self.enc = tok(texts, truncation=True, max_length=maxlen, padding=False)["input_ids"]
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.enc[i], self.y[i]


def collate(batch, pad_id):
    mx = max(len(b[0]) for b in batch)
    ids = torch.full((len(batch), mx), pad_id, dtype=torch.long)
    am = torch.zeros((len(batch), mx), dtype=torch.long)
    for i, (seq, _) in enumerate(batch):
        ids[i, :len(seq)] = torch.tensor(seq)
        am[i, :len(seq)] = 1
    return ids, am, torch.tensor([b[1] for b in batch])


class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        # fp32 master weights: AdamW on bf16 params cannot resolve a 3e-5 update
        self.enc = AutoModel.from_pretrained(BASE, dtype=torch.float32)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(h, n)

    def forward(self, ids, am):
        o = self.enc(input_ids=ids, attention_mask=am)
        v = o.last_hidden_state[:, 0]
        return self.fc(self.drop(v.float()))


def pick_stage1(kind, train, hold, only="", clean=False):
    uni = pd.read_parquet("extra/union_all.parquet")
    if only:
        uni = uni[uni.source == only]
    pool = uni[uni.in_pool_v1]
    marg = uni[(~uni.in_train) & (~uni.in_pool_v1)]
    bad = set(hold["text"].str.lower().str.strip()) | set(train["text"].str.lower().str.strip())
    pool = pool[~pool["text"].str.lower().str.strip().isin(bad)]
    marg = marg[~marg["text"].str.lower().str.strip().isin(bad)]
    if clean:
        from invariants import rules_from_train, violations
        rules = rules_from_train()
        pool = pool[~violations(pool, rules)]
        marg = marg[~violations(marg, rules)]
    return {"pool": pool, "marg": marg,
            "pool_marg": pd.concat([pool, marg], ignore_index=True)}[kind]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="none", choices=["none", "pool", "marg", "pool_marg"])
    ap.add_argument("--only", default="", help="keep only this source tag in stage 1")
    ap.add_argument("--clean", action="store_true", help="drop invariant-violating rows from stage 1")
    ap.add_argument("--drop", default="", help="comma list of source tags to drop from stage 1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--ep1", type=int, default=1)
    ap.add_argument("--ep2", type=int, default=3)
    ap.add_argument("--lr1", type=float, default=3e-5)
    ap.add_argument("--lr2", type=float, default=2e-5)
    ap.add_argument("--cap", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit2", type=int, default=0)
    ap.add_argument("--eval-each", action="store_true")
    a = ap.parse_args()
    tag = a.tag or a.stage1 or "none"

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    tr = pd.read_csv("data/train.csv")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    i1, i2 = next(gss.split(tr, tr["model"], groups=tr["prompt_id"]))
    train, hold = tr.iloc[i1].reset_index(drop=True), tr.iloc[i2].reset_index(drop=True)
    if a.limit2:
        train = train.sample(min(a.limit2, len(train)), random_state=0).reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    pad = tok.pad_token_id or 0
    net = Net(len(MODELS)).cuda()
    print(f"train {len(train)} | holdout {len(hold)}", flush=True)

    def evaluate():
        net.eval()
        ds = DS(hold["text"].tolist(), [L2I[m] for m in hold["model"]], tok, a.maxlen)
        dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=lambda b: collate(b, pad))
        pr = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for step, (ids, am, y) in enumerate(dl):
                pr.append(net(ids.cuda(), am.cuda()).float().argmax(-1).cpu().numpy())
        p = np.concatenate(pr)
        y = np.array([L2I[m] for m in hold["model"]])
        net.train()
        return p, y

    def stage(df, epochs, lr, name):
        ds = DS(df["text"].tolist(), [L2I[m] for m in df["model"]], tok, a.maxlen)
        dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=0,
                        collate_fn=lambda b: collate(b, pad))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
        steps = max(1, epochs * len(dl))
        from transformers import get_cosine_schedule_with_warmup
        sch = get_cosine_schedule_with_warmup(opt, max(1, steps // 20), steps)
        net.train(); t0 = time.time(); tot = n = 0
        for ep in range(epochs):
            tot = n = 0
            for step, (ids, am, y) in enumerate(dl):
                ids, am, y = ids.cuda(), am.cuda(), y.cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = net(ids, am)
                loss = F.cross_entropy(logits, y, label_smoothing=0.05)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
                tot += loss.item() * len(y); n += len(y)
                if step % 25 == 0:
                    el = time.time() - t0
                    print(f"  [{name}] ep{ep+1} step{step}/{len(dl)} loss={loss.item():.4f} "
                          f"{el:.0f}s ({n/el:.0f} samples/s)", flush=True)
            msg = f"[{name}] ep{ep+1}/{epochs} loss={tot/n:.4f} lr={sch.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s"
            if a.eval_each:
                p_, y_ = evaluate()
                msg += f" hold_bal_acc={balanced_accuracy_score(y_, p_):.4f}"
            print(msg, flush=True)

    if a.stage1 != "none":
        s1 = pick_stage1(a.stage1, train, hold, a.only, a.clean)
        if a.drop:
            s1 = s1[~s1.source.isin(a.drop.split(","))]
        per = max(1, a.cap // len(MODELS))
        # NB: groupby().apply() drops the grouping column on pandas >= 2.2 -- use head() instead
        s1 = (s1.sample(frac=1.0, random_state=a.seed)
                .groupby("model", sort=False).head(per)
                .reset_index(drop=True))
        print(f"stage1 rows={len(s1)} sources={s1.source.nunique()}", flush=True)
        stage(s1, a.ep1, a.lr1, "s1")

    stage(train, a.ep2, a.lr2, "s2")

    p, y = evaluate()
    res = dict(tag=tag, stage1=a.stage1, only=a.only, drop=a.drop, clean=bool(a.clean), n_s1=0 if a.stage1 == "none" else len(s1),
               bal_acc=float(balanced_accuracy_score(y, p)), macro_f1=float(f1_score(y, p, average="macro")))
    print("RESULT " + json.dumps(res), flush=True)
    os.makedirs("extra/proxy", exist_ok=True)
    json.dump(res, open(f"extra/proxy/{tag}.json", "w"), indent=1)
    np.save(f"extra/proxy/{tag}_pred.npy", p)


if __name__ == "__main__":
    main()
