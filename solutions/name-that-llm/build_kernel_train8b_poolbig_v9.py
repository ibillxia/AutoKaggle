"""Build ntl-train8b-poolbig-v9.ipynb: train8b_v2 recipe unchanged, only the stage-1 pool is swapped
for pool_big_v2 = pool_v2 (70,034) merged with crtproduction's OPEN POOL v9 (192,926 CLEANED rows,
discussion 752766: raw-vs-clean same-scale experiment gave +0.027; ~25% of the old open pool was
junk/off-genre). poolbig v1 (dirty pool) holdout = 0.5389; expect the clean pool to add on top.
"""
import json
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train8b_poolbig_v9")
os.makedirs(OUT_DIR, exist_ok=True)

SETUP = r'''
import gc, glob, json, math, os, random, time

# Kaggle preinstalls torchao 0.10.0; recent peft raises ImportError during LoRA
# injection when it sees an unsupported torchao. Uninstall first (offline-safe).
os.system("pip uninstall -y torchao >/dev/null 2>&1")
# Expandable segments reduce fragmentation with big fp16 weights on 15GB T4s.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from sklearn.metrics import balanced_accuracy_score

T0 = time.time()
INPUT_DIR = "/kaggle/input"
KEEP, LAYERS, H = 24, [2, 4, 6, 8, 12, 16, 20, 24], 4096
MAX_LEN = 448
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}
TRIPLETS_PER_STEP, BS = 4, 3          # 4 triplets x 3 passages = 12 texts per step (T4 OOM headroom)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def find(pattern):
    hits = sorted(glob.glob(f"{INPUT_DIR}/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]

def find_base(spec):
    # weight-aware (same fix as hybrid4 v2): a matching config.json must sit next to real weights
    fallback = None
    for cfgp in sorted(glob.glob(f"{INPUT_DIR}/**/config.json", recursive=True)):
        try:
            c = json.load(open(cfgp))
        except Exception:
            continue
        if all(c.get(k) == v for k, v in spec.items()):
            d = os.path.dirname(cfgp)
            if os.path.exists(os.path.join(d, "model.safetensors")) or os.path.exists(os.path.join(d, "pytorch_model.bin")) \
               or glob.glob(os.path.join(d, "model-*.safetensors")) or glob.glob(os.path.join(d, "pytorch_model-*.bin")):
                return d
            if fallback is None:
                fallback = d
    if fallback is not None:
        return fallback
    raise FileNotFoundError(f"base model {spec}")

_competition_dir = next(p for p in ["/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm"] if os.path.exists(p + "/train.csv"))
train = pd.read_csv(_competition_dir + "/train.csv")
train["text"] = train["text"].fillna("").astype(str)
y_train = train["model"].map(LAB2I).values
print("train", train.shape, "gpus", torch.cuda.device_count(), flush=True)

BASE_DIR = find_base(dict(model_type="qwen3", hidden_size=H, num_hidden_layers=36))
print("base:", BASE_DIR, flush=True)
'''

POOL = r'''
# ---------------- merged stage-1 pool: pool_v2 + crtproduction open pool ----------------
p2 = pd.read_parquet(find("pool_v2.parquet"))                      # 70,034 rows (our v2 pool)
opf = find("ntl_open_pool.parquet")
op = pd.read_parquet(opf)                                          # 192,926 rows (CLEANED open pool v9)
print("open pool", op.shape, opf, flush=True)
# v9 is already cleaned by the author (discussion 752766: +0.027 raw->clean at same scale).
# We only drop the gemini-3.7 routing-risk rows; kb_proxy rows are KEPT (removing them was a wash).
op = op[~op["flag_gemini37_routing_risk"] & ~op["flag_near_train_prompt"] & ~op["flag_train_prompt_regen"]]
print("after safety filters:", len(op), flush=True)
# cross-dedup by whitespace-normalized text hash (pool_v2 wins)
def thash(t):
    return hashlib.md5(" ".join(str(t).split()).encode()).hexdigest()
h2 = set(p2["text"].map(thash))
oh = op["text"].map(thash)
dup = oh.isin(h2)
print("open-pool rows already in pool_v2:", int(dup.sum()), flush=True)
op = op[~dup][["text", "model", "prompt_group"]].rename(columns={"prompt_group": "prompt_id"})
pool = pd.concat([p2[["text", "model", "prompt_id"]], op], ignore_index=True)
pool = pool.drop_duplicates(subset=["text", "model"]).reset_index(drop=True)
pool["text"] = pool["text"].fillna("").astype(str)
pool = pool[pool["model"].isin(LABELS)].reset_index(drop=True)
y_pool = pool["model"].map(LAB2I).values
assert y_pool.min() >= 0 and not np.isnan(y_pool).any()
usable = pool.groupby("prompt_id")["text"].transform("size") >= BS
print("pool_big_v2:", pool.shape, "| unique prompts:", pool["prompt_id"].nunique(),
      "| rows in >=3-prompts:", int(usable.sum()), flush=True)
print(pool["model"].value_counts(), flush=True)
pool.to_parquet("/kaggle/working/pool_big_v9.parquet", index=False)
'''

MODEL = r'''
# ---------------- model / LoRA ----------------
tok = AutoTokenizer.from_pretrained(BASE_DIR)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mc = AutoConfig.from_pretrained(BASE_DIR)
mc.num_hidden_layers, mc.use_cache = KEEP, False
if getattr(mc, "layer_types", None) is not None:
    mc.layer_types = mc.layer_types[:KEEP]
n_gpu = torch.cuda.device_count()
# 24/36 layers of 8B fp16 is ~10.6GB of weights; cap each GPU's share so
# device_map splits layers across both T4s and leaves room for activations.
per_gpu = "6GiB" if n_gpu >= 2 else "13GiB"
mem = {i: per_gpu for i in range(n_gpu)}
model = AutoModel.from_pretrained(BASE_DIR, config=mc, dtype=torch.float16, device_map="auto", max_memory=mem).eval()
model.norm = nn.Identity()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                  bias="none", task_type=None)
model = get_peft_model(model, lcfg)
# IMPORTANT: enable gradient checkpointing AFTER the peft wrap — calling it on
# the bare base model is silently lost once peft re-wraps the modules.
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()
model.print_trainable_parameters()

norms = nn.ModuleList([nn.LayerNorm(H) for _ in LAYERS]).float()
head_plain = nn.Linear(H * len(LAYERS), K).float()
head_centred = nn.Linear(H * len(LAYERS), K).float()
dev0 = "cuda:0"
norms, head_plain, head_centred = norms.to(dev0), head_plain.to(dev0), head_centred.to(dev0)

@torch.no_grad()
def extract(texts, bs=8):
    """Pooled features Z (n, 8*H): mean over non-pad tokens minus position 0, LayerNorm after pooling."""
    model.eval()
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=MAX_LEN)["input_ids"]
    ids = [x + [tok.eos_token_id] for x in ids]
    order = np.argsort([len(x) for x in ids], kind="stable")
    Z = np.zeros((len(ids), H * len(LAYERS)), np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(model.embed_tokens.weight.device)
        with torch.autocast("cuda", torch.float16):
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float().to(dev0)
        m[:, 0] = 0
        den = m.sum(1).clamp(min=1)
        Z[idx] = torch.cat([nm((hs[l].to(dev0).float() * m).sum(1) / den) for l, nm in zip(LAYERS, norms)], -1).cpu().numpy()
    model.train()
    return Z

def make_batches(df, y, epochs, rng, full_coverage=True):
    """Yield lists of (text, label) of size TRIPLETS_PER_STEP*BS: triplets share a prompt.

    full_coverage=True draws round(len(rows)/BS) triplets per prompt so one epoch actually
    visits every passage (pool has ~15.5 passages/prompt, so the old one-triplet version only
    saw 19% of the pool and left stage1 badly undertrained).
    """
    by_prompt = {}
    for pid, t, yy in zip(df["prompt_id"].values, df["text"].values, y):
        by_prompt.setdefault(pid, []).append((t, yy))
    pids = list(by_prompt)
    for _ in range(epochs):
        rng.shuffle(pids)
        buf = []
        for pid in pids:
            rows = by_prompt[pid]
            if len(rows) < BS:
                continue
            n_trip = max(1, round(len(rows) / BS)) if full_coverage else 1
            for _ in range(n_trip):
                pick = rng.sample(rows, BS)
                models = {yy for _, yy in pick}
                if len(models) < 2:      # need at least 2 classes to be informative
                    continue
                buf.append(pick)
                if len(buf) == TRIPLETS_PER_STEP:
                    yield [x for tri in buf for x in tri]
                    buf = []

def run_epoch(batches, opt, sched, scaler, cw, hook=None, hook_every=0):
    tot, nsteps, nan_skips = 0.0, 0, 0
    for batch in batches:
        texts = [t for t, _ in batch]
        yy = torch.tensor([l for _, l in batch], device=dev0)
        with torch.autocast("cuda", torch.float16):
            ids = tok(texts, add_special_tokens=False, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(model.embed_tokens.weight.device)
            hs = model(**ids, output_hidden_states=True, use_cache=False).hidden_states
        m = ids["attention_mask"].unsqueeze(-1).float().to(dev0)
        m[:, 0] = 0
        den = m.sum(1).clamp(min=1)
        feats = torch.cat([nm((hs[l].to(dev0).float() * m).sum(1) / den) for l, nm in zip(LAYERS, norms)], -1)
        logits = head_plain(feats)
        loss = F.cross_entropy(logits, yy, weight=cw)
        # centred head on triplet-centred features
        f3 = feats.view(TRIPLETS_PER_STEP, BS, -1)
        fc = f3 - f3.mean(1, keepdim=True)
        logits_c = head_centred(fc.reshape(-1, fc.shape[-1]))
        loss = loss + F.cross_entropy(logits_c, yy, weight=cw)
        if not torch.isfinite(loss):
            nan_skips += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot += float(loss); nsteps += 1
        if hook is not None and hook_every and nsteps % hook_every == 0:
            hook(nsteps)
        if nsteps % 100 == 0:
            print(f"  step {nsteps} loss {tot/nsteps:.4f} ({time.time()-T0:.0f}s)", flush=True)
    return tot / max(nsteps, 1), nan_skips

def save_ckpt(out_dir, note):
    os.makedirs(os.path.join(out_dir, "adapter"), exist_ok=True)
    model.save_pretrained(os.path.join(out_dir, "adapter"))
    sd = {}
    for i, nm in enumerate(norms):
        sd[f"norms.{i}.weight"] = nm.weight.data.clone().cpu()
        sd[f"norms.{i}.bias"] = nm.bias.data.clone().cpu()
    sd["plain.weight"] = head_plain.weight.data.clone().cpu()
    sd["plain.bias"] = head_plain.bias.data.clone().cpu()
    sd["centered.weight"] = head_centred.weight.data.clone().cpu()
    sd["centered.bias"] = head_centred.bias.data.clone().cpu()
    save_file(sd, os.path.join(out_dir, "heads.safetensors"))
    json.dump({"labels": LABELS, "note": note,
               "cfg": {"keep": KEEP, "layers": LAYERS, "hidden": H, "max_len": MAX_LEN,
                       "model_type": "qwen3", "num_hidden_layers": 36}},
              open(os.path.join(out_dir, "meta.json"), "w"))
    print("saved", out_dir, flush=True)
'''

STAGE1 = r'''
# ---------------- stage 1: pool_big ----------------
# Full epoch on pool_big (~259k rows, ~21.5k steps at 12 texts/step) is impossible inside a
# 9h session (~7s/step); pool_v2's run used 1900 steps (~3.8h). STEP_BUDGET=3300 (~6.4h) keeps
# the whole kernel ~7.9h < 9h cap. A partial-epoch budget sees MORE unique passages than the
# full pool_v2 epoch (40k vs 23k visits) and far more prompt diversity. Checkpoints every 400
# steps so a late failure is not total.
STEP_BUDGET = 3300


def capped(gen, limit):
    n = 0
    for b in gen:
        if limit and n >= limit:
            return
        n += 1
        yield b


rng = random.Random(SEED)
cnt = np.bincount(y_pool, minlength=K).astype(np.float64)
cw = torch.tensor((len(y_pool) / (K * np.maximum(cnt, 1))), dtype=torch.float32, device=dev0)
steps1 = min(STEP_BUDGET, sum(1 for _ in make_batches(pool, y_pool, 1, random.Random(SEED))))
print("stage1 steps:", steps1, flush=True)
opt = torch.optim.AdamW([
    dict(params=[p for n, p in model.named_parameters() if p.requires_grad], lr=1e-4),
    dict(params=list(norms.parameters()) + list(head_plain.parameters()) + list(head_centred.parameters()), lr=1e-3)])
scaler = torch.amp.GradScaler("cuda")
sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps1), steps1)
t1 = time.time()
loss1, nans = run_epoch(capped(make_batches(pool, y_pool, 1, rng), STEP_BUDGET), opt, sched, scaler, cw,
                        hook=lambda n: save_ckpt("/kaggle/working/stage1", f"stage1 partial {n} steps"),
                        hook_every=400)
print(f"stage1 done: loss {loss1:.4f} nan_skips {nans} ({time.time()-t1:.0f}s)", flush=True)
save_ckpt("/kaggle/working/stage1", f"stage1 pool_big {steps1} steps")
'''

STAGE2 = r'''
# ---------------- stage 2: official ----------------
def stage2(fit_df, fit_y, epochs, lr_lora, lr_head, tag):
    rng = random.Random(SEED + 1)
    cnt = np.bincount(fit_y, minlength=K).astype(np.float64)
    cw = torch.tensor((len(fit_y) / (K * np.maximum(cnt, 1))), dtype=torch.float32, device=dev0)
    steps = sum(1 for _ in make_batches(fit_df, fit_y, epochs, random.Random(SEED + 1)))
    opt = torch.optim.AdamW([
        dict(params=[p for n, p in model.named_parameters() if p.requires_grad], lr=lr_lora),
        dict(params=list(norms.parameters()) + list(head_plain.parameters()) + list(head_centred.parameters()), lr=lr_head)])
    scaler = torch.amp.GradScaler("cuda")
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    run_epoch(make_batches(fit_df, fit_y, epochs, rng), opt, sched, scaler, cw)
    print(f"{tag} trained ({time.time()-T0:.0f}s)", flush=True)

# --- 2a: 120-prompt fit / 30-prompt holdout for honest CV ---
pids = sorted(train["prompt_id"].unique())
rng = random.Random(7); rng.shuffle(pids)
hold_pids, fit_pids = set(pids[:30]), set(pids[30:])
fit_df = train[train["prompt_id"].isin(fit_pids)].reset_index(drop=True)
hold_df = train[train["prompt_id"].isin(hold_pids)].reset_index(drop=True)
stage2(fit_df, fit_df["model"].map(LAB2I).values, 2, 5e-5, 5e-4, "stage2a")
Zh = extract(hold_df["text"].tolist())
pred = np.array(LABELS)[head_plain(torch.tensor(Zh, device=dev0)).argmax(1).cpu().numpy()]
acc = balanced_accuracy_score(hold_df["model"].values, pred)
print(f"HOLDOUT (30 prompts) balanced acc after stage2a: {acc:.4f}  (pool_v2 0.4852 / poolbig-dirty 0.5389)", flush=True)
with open("/kaggle/working/cv_report.txt", "w") as f:
    f.write(f"stage2a_holdout_balanced_acc\t{acc:.6f}\n")
save_ckpt("/kaggle/working/ntl8b_poolbig9_cv", f"stage2a 120 prompts, holdout {acc:.4f}")

# --- 2b: reload stage1, train on all 150 prompts for the submission checkpoint ---
from safetensors.torch import load_file
sd = load_file("/kaggle/working/stage1/heads.safetensors")
norms.load_state_dict({k[len('norms.'):]: v for k, v in sd.items() if k.startswith('norms.')})
head_plain.load_state_dict({k[len('plain.'):]: v for k, v in sd.items() if k.startswith('plain.')})
head_centred.load_state_dict({k[len('centered.'):]: v for k, v in sd.items() if k.startswith('centered.')})
stage2(train, y_train, 2, 5e-5, 5e-4, "stage2b")
save_ckpt("/kaggle/working/ntl8b_poolbig9_v1", "stage2b all 150 prompts")
print(f"ALL DONE in {time.time()-T0:.0f}s", flush=True)
'''

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "# Two-stage LoRA, Qwen3-8B-Base L24 — pool_big edition\n",
        "Identical recipe to ntl-train8b-v2 (validated, stage2a holdout 0.4852); ONLY the stage-1 pool changes:\n",
        "pool_big = pool_v2 (70,034) + crtproduction open pool (254,561, safety-filtered, cross-deduped) ~ 259k rows.\n",
        "Evidence: same-recipe pool upsize 36k->124k gave CV 0.578->0.611, LB 0.586->0.617 (discussion 752810).\n",
        "Stage 1: up to 3300 steps on pool_big (class-weighted CE, same-prompt triplets, ckpt every 400 steps).\n",
        "Stage 2a: 2 epochs on 120 official prompts + holdout eval on 30 prompts.\n",
        "Stage 2b: reload stage1, 2 epochs on all 150 prompts -> submission checkpoint."]},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": SETUP},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": POOL},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": MODEL},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": STAGE1},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": STAGE2},
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}

with open(os.path.join(OUT_DIR, "ntl-train8b-poolbig-v9.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

meta = {
    "id": "ibillxia/ntl-train8b-poolbig-v9",
    "title": "ntl train8b poolbig v9",
    "code_file": "ntl-train8b-poolbig-v9.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "competition_sources": ["name-that-llm"],
    "dataset_sources": ["ibillxia/ntl-pool-v2", "crtproduction/name-that-llm-open-pool"],
    "model_sources": ["qwen-lm/qwen-3/transformers/8b-base/1"],
}
with open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("written to", OUT_DIR)
