# Build a Kaggle training kernel: 4B fp16 LoRA, 18-way head, prompt-disjoint holdout.
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "ntl_train4b.ipynb"
MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = [
    md("""# 4B LoRA fine-tune (18-way, prompt-disjoint holdout)

Kaggle has no bitsandbytes, so this is full fp16 with LoRA on all linear layers.
Holdout = 30 prompts never seen in training; the score printed each epoch is the honest gate."""),

    code("""\
import glob, os, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit

MODELS = """ + json.dumps(MODELS) + """
BS, ACCUM, LR, HEAD_LR, LORA_R, MAXLEN, EPOCHS = 4, 2, 2e-4, 1e-3, 16, 384, 3

def find(name):
    for pat in (f"/kaggle/input/competitions/name-that-llm/{name}", f"/kaggle/input/name-that-llm/{name}"):
        if os.path.exists(pat):
            return pat
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, name
    return hits[0]

def find_dir(*names):
    for pat in ("/kaggle/input/{}", "/kaggle/input/datasets/*/{}"):
        for n in names:
            for hit in glob.glob(pat.format(n)):
                if os.path.isdir(hit) and any(f.endswith("index.json") for f in os.listdir(hit)):
                    return hit
    raise FileNotFoundError(names)

df = pd.read_csv(find("train.csv"))
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
a, b = next(gss.split(df, df["model"], groups=df["prompt_id"]))
train, hold = df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True)
y_tr = np.array([MODELS.index(m) for m in train["model"]])
y_ho = np.array([MODELS.index(m) for m in hold["model"]])
print(len(train), len(hold), torch.cuda.device_count(), "gpus")"""),

    code("""\
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

DS = find_dir("ntl-qwen4b")
tok = AutoTokenizer.from_pretrained(DS)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base = AutoModel.from_pretrained(DS, dtype=torch.float16, device_map={"": 0}).eval()
base.gradient_checkpointing_enable()
base.enable_input_require_grads()
base = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
                                       bias="none", target_modules="all-linear"))
base.print_trainable_parameters()
HIDDEN = base.config.text_config.hidden_size


class Head(nn.Module):
    def __init__(self, body, hidden, n_cls, p=0.1):
        super().__init__()
        self.base, self.drop, self.fc = body, nn.Dropout(p), nn.Linear(hidden, n_cls)

    def forward(self, ids, am):
        hs = self.base(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).to(hs.dtype)
        return self.fc(self.drop(((hs * m).sum(1) / m.sum(1).clamp_min(1)).float()))


model = Head(base, HIDDEN, len(MODELS)).cuda()
opt = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if n.startswith("fc")], "lr": HEAD_LR},
    {"params": [p for n, p in model.named_parameters() if not n.startswith("fc") and p.requires_grad],
     "lr": LR}], weight_decay=0.01)
steps = (len(train) + BS * ACCUM - 1) // (BS * ACCUM) * EPOCHS
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[HEAD_LR, LR], total_steps=steps, pct_start=0.05)
scaler = torch.amp.GradScaler("cuda")


def encode(texts):
    e = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
    return e["input_ids"].cuda(), e["attention_mask"].cuda()


@torch.no_grad()
def predict(texts, bs=16):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        ids, am = encode(texts[i:i + bs])
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.softmax(model(ids, am).float(), -1).cpu().numpy())
    model.train()
    return np.concatenate(out)


texts = train["text"].tolist()
rng = np.random.default_rng(0)
best = -1.0
for ep in range(EPOCHS):
    t0, run = time.time(), 0.0
    order = rng.permutation(len(train))
    model.train()
    for i in range(0, len(order), BS):
        idx = order[i:i + BS]
        ids, am = encode([texts[j] for j in idx])
        y = torch.tensor(y_tr[idx]).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = nn.functional.cross_entropy(model(ids, am), y)
        scaler.scale(loss / ACCUM).backward()
        if (i // BS) % ACCUM == ACCUM - 1:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
        run += float(loss)
        if (i // BS) % 25 == 0:
            el = time.time() - t0
            print(f"ep{ep} {i}/{len(order)} loss {run / (i / BS + 1):.4f} {el / (i // BS + 1) * 1000:.0f} ms/step", flush=True)
    P = predict(hold["text"].tolist())
    acc = balanced_accuracy_score(hold["model"].values, [MODELS[i] for i in P.argmax(1)])
    print(f"== ep{ep} holdout bal_acc {acc:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if acc > best:
        best = acc
        model.base.save_pretrained(f"/kaggle/working/adapter_ep{ep}")
        torch.save(model.fc.state_dict(), f"/kaggle/working/head_ep{ep}.pt")
        np.savez(f"/kaggle/working/holdout_probs_ep{ep}.npz", probs=P, y=hold["model"].values)
print("best", best)"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}

if __name__ == "__main__":
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": "ibillxia/ntl-train4b", "title": "ntl train4b", "code_file": OUT.name,
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": False,
            "dataset_sources": ["ibillxia/ntl-qwen4b"],
            "competition_sources": ["name-that-llm"], "kernel_sources": []}
    (HERE / "kernel-metadata-train4b.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", OUT)
