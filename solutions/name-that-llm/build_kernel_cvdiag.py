"""Build ntl-cvdiag.ipynb: honest 5-fold (grouped by prompt) CV of the public two-stage recipe.

Why: the author's markdown claims CV 0.593 / LB 0.592 for the full blend, while our fork of the
same code+weights scored LB 0.576. This kernel answers two questions at once:
  1) what is the honest CV of the public pipeline (n-gram refit per fold, LoRA frozen)?
  2) how much does each "trick" (centred twin, Sinkhorn) actually add?
It also fits a deliberately leaky n-gram (full train+pool) to quantify how much a naive CV inflates.
"""
import json
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cvdiag")
os.makedirs(OUT_DIR, exist_ok=True)

SETUP = r'''
import gc, glob, json, os, re, time
import numpy as np
import pandas as pd
import torch
from torch import nn

T0 = time.time()
INPUT_DIR = "/kaggle/input"

W_SVC = 1.021
LORAS = {
    "lora8b_v1":  dict(w=0.226, twin_w=0.079, bs=16, base=dict(model_type="qwen3", hidden_size=4096, num_hidden_layers=36)),
    "lora14b_v1": dict(w=0.429, twin_w=0.150, bs=8,  base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),
}
TWIN = dict(thresh=0.85, min_size=2, max_size=18)

LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}


def find(pattern):
    hits = sorted(glob.glob(f"{INPUT_DIR}/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


train = pd.read_csv(find("train.csv"))
train["text"] = train["text"].fillna("").astype(str)
y = train["model"].map(LAB2I).values
pool = pd.read_parquet(find("pool_v1.parquet"))
pool["text"] = pool["text"].fillna("").astype(str)
print("train", train.shape, "pool", pool.shape, "prompts", train.prompt_id.nunique(), flush=True)
'''

LORA = r'''
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer


def find_base(spec):
    for cfg in sorted(glob.glob(f"{INPUT_DIR}/**/config.json", recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if all(c.get(k) == v for k, v in spec.items()):
            return os.path.dirname(cfg)
    raise FileNotFoundError(f"attach the base model {spec}")


def load_trunk(base_dir, keep, dtype):
    mc = AutoConfig.from_pretrained(base_dir)
    mc.num_hidden_layers, mc.use_cache = keep, False
    if getattr(mc, "layer_types", None) is not None:
        mc.layer_types = mc.layer_types[:keep]
    hd = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)
    gib = 2 * (mc.vocab_size * mc.hidden_size + keep * (2 * mc.hidden_size * mc.num_attention_heads * hd
               + 2 * mc.hidden_size * mc.num_key_value_heads * hd + 3 * mc.hidden_size * mc.intermediate_size)) / 2 ** 30
    if gib > 13 and torch.cuda.device_count() > 1:
        mem = {i: "13GiB" for i in range(torch.cuda.device_count())}
        model = AutoModel.from_pretrained(base_dir, config=mc, dtype=dtype, device_map="auto", max_memory=mem).eval()
    else:
        model = AutoModel.from_pretrained(base_dir, config=mc, dtype=dtype).to("cuda:0").eval()
    model.norm = nn.Identity()
    return model


@torch.no_grad()
def merge_lora(model, adapter_dir):
    acfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scale = acfg["lora_alpha"] / acfg["r"]
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    modules = dict(model.named_modules())
    for k, A in sd.items():
        if ".lora_A." in k:
            name = k.split(".lora_A.")[0]
            lin = modules[name[name.index("layers."):]]
            Bm = sd[k.replace(".lora_A.", ".lora_B.")]
            lin.weight.add_(((Bm.float().to(lin.weight.device) @ A.float().to(lin.weight.device)) * scale).to(lin.weight.dtype))


@torch.no_grad()
def lora_features(ckpt_dir, base_dir, texts, bs):
    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
    assert meta["labels"] == LABELS, "label order differs"
    cfg = meta["cfg"]
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    tok = AutoTokenizer.from_pretrained(base_dir)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_trunk(base_dir, cfg["keep"], dtype)
    merge_lora(model, os.path.join(ckpt_dir, "adapter"))
    heads = load_file(os.path.join(ckpt_dir, "heads.safetensors"))
    layers, H = cfg["layers"], model.config.hidden_size
    norms = nn.ModuleList([nn.LayerNorm(H) for _ in layers]).to("cuda:0").float()
    norms.load_state_dict({k[len("norms."):]: v for k, v in heads.items() if k.startswith("norms.")})
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=cfg["max_len"])["input_ids"]
    ids = [x + [tok.eos_token_id] for x in ids]
    order = np.argsort([len(x) for x in ids], kind="stable")
    Z = np.zeros((len(ids), H * len(layers)), np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(model.embed_tokens.weight.device)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        if not all(torch.isfinite(hs[l]).all() for l in layers):
            model.to(torch.bfloat16)
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            model.to(dtype)
        m = enc["attention_mask"].unsqueeze(-1).float().to("cuda:0")
        m[:, 0] = 0
        den = m.sum(1).clamp(min=1)
        Z[idx] = torch.cat([nm((hs[l].to("cuda:0").float() * m).sum(1) / den) for l, nm in zip(layers, norms)], -1).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return Z, {k: v.float().numpy() for k, v in heads.items() if not k.startswith("norms.")}


FEATS = {}
for name, mc in LORAS.items():
    ckpt = os.path.dirname(find(f"{name}/meta.json"))
    Z, heads = lora_features(ckpt, find_base(mc["base"]), train["text"].tolist(), mc["bs"])
    FEATS[name] = (Z, heads)
    print(f"{name} features {Z.shape} ({time.time()-T0:.0f}s)", flush=True)
    np.savez_compressed(f"/kaggle/working/feat_{name}.npz", Z=Z, **{k: v for k, v in heads.items()})
'''

CV = r'''
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import balanced_accuracy_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]


def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def function_text(t):
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))


def fit_svc(fit_texts, fit_y, pred_texts):
    A, B = [], []
    for view, lo, hi, max_features, weight in VIEWS:
        prep = shape_text if view == "shape" else function_text if view == "function" else (lambda s: s)
        analyzer = "char" if view == "shape" else "word" if view == "function" else view
        opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                    max_features=max_features, dtype=np.float32)
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vec = TfidfVectorizer(**opts)
        A.append(vec.fit_transform([prep(t) for t in fit_texts]) * weight)
        B.append(vec.transform([prep(t) for t in pred_texts]) * weight)
    svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced").fit(hstack(A, format="csr"), fit_y)
    P = softmax(svc.decision_function(hstack(B, format="csr")))
    del A, B, svc
    gc.collect()
    return P


def twin_terms(Z, heads, groups, twin_rows, twin_w, logit):
    if twin_rows.any():
        Zc = Z.copy()
        for g in np.unique(groups[twin_rows]):
            Zc[groups == g] -= Zc[groups == g].mean(0)
        centred = Zc[twin_rows] @ heads["centered.weight"].T + heads["centered.bias"]
        logit[twin_rows] += twin_w * np.log(np.clip(softmax(centred), 1e-9, 1))
    return logit


def cluster(texts):
    Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(list(texts))
    D = np.clip(1 - (Xt @ Xt.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    groups = fcluster(linkage(squareform(D, checks=False), method="average"), TWIN["thresh"], criterion="distance") \
        if len(texts) >= 2 else np.ones(len(texts), dtype=int)
    size = np.bincount(groups)[groups]
    twin_rows = (size >= TWIN["min_size"]) & (size <= TWIN["max_size"])
    return groups, twin_rows


def sinkhorn(P):
    P = P.copy()
    target = np.full(K, len(P) / K)
    for _ in range(100):
        P *= target / P.sum(0)
        P /= P.sum(1, keepdims=True)
    return P


# ---- deliberately leaky n-gram: fitted on ALL of train+pool (what a naive CV would do) ----
pool_y = pool["model"].map(LAB2I).values
P_svc_leaky = fit_svc(train["text"].tolist() + pool["text"].tolist(), np.r_[y, pool_y], train["text"].tolist())
print(f"leaky n-gram done ({time.time()-T0:.0f}s)", flush=True)

# ---- 5 folds grouped by prompt ----
pids = sorted(train["prompt_id"].unique())
rng = np.random.RandomState(2026)
rng.shuffle(pids)
fold_of_prompt = {p: i % 5 for i, p in enumerate(pids)}
fold_id = train["prompt_id"].map(fold_of_prompt).values
print("fold sizes:", np.bincount(fold_id), flush=True)

VARIANTS = ["A_svc_only", "B_svc+lora_plain", "C_+twin", "D_+twin+sinkhorn",
            "E_svc_leaky+lora_plain", "F_leaky+twin+sinkhorn",
            "G_lora8b_plain", "H_lora14b_plain"]
scores = {v: [] for v in VARIANTS}
twin_frac = []

for f in range(5):
    va = fold_id == f
    tr = ~va
    P_svc = fit_svc(train.loc[tr, "text"].tolist() + pool["text"].tolist(),
                    np.r_[y[tr], pool_y], train.loc[va, "text"].tolist())
    print(f"fold {f}: honest n-gram done ({time.time()-T0:.0f}s), val={int(va.sum())}", flush=True)
    yv = y[va]
    texts_va = train.loc[va, "text"].tolist()
    groups, twin_rows = cluster(texts_va)
    twin_frac.append(float(twin_rows.mean()))
    base = W_SVC * np.log(np.clip(P_svc, 1e-9, 1))
    leaky = W_SVC * np.log(np.clip(P_svc_leaky[va], 1e-9, 1))
    lora_plain = np.zeros((int(va.sum()), K))
    lora_twin = np.zeros((int(va.sum()), K))
    per_model_plain = {}
    for name, mc in LORAS.items():
        Z, heads = FEATS[name]
        Zv = Z[va]
        plain = softmax(Zv @ heads["plain.weight"].T + heads["plain.bias"])
        lp = mc["w"] * np.log(np.clip(plain, 1e-9, 1))
        lora_plain += lp
        per_model_plain[name] = lp
        lora_twin = twin_terms(Zv, heads, groups, twin_rows, mc["twin_w"], lora_twin)
    L = base + lora_plain
    LOGITS = {
        "A_svc_only": base,
        "B_svc+lora_plain": L,
        "C_+twin": L + lora_twin,
        "D_+twin+sinkhorn": None,        # sinkhorn applied below
        "E_svc_leaky+lora_plain": leaky + lora_plain,
        "F_leaky+twin+sinkhorn": None,
        "G_lora8b_plain": base + per_model_plain["lora8b_v1"],
        "H_lora14b_plain": base + per_model_plain["lora14b_v1"],
    }
    for v, lg in LOGITS.items():
        if lg is None:
            continue
        scores[v].append(balanced_accuracy_score(yv, softmax(lg).argmax(1)))
    scores["D_+twin+sinkhorn"].append(balanced_accuracy_score(yv, sinkhorn(softmax(L + lora_twin)).argmax(1)))
    scores["F_leaky+twin+sinkhorn"].append(balanced_accuracy_score(yv, sinkhorn(softmax(leaky + lora_plain + lora_twin)).argmax(1)))
    print("  " + "  ".join(f"{v}={scores[v][-1]:.4f}" for v in VARIANTS), flush=True)

L = []
L.append("honest 5-fold CV, grouped by prompt_id (n-gram refit per fold, LoRA frozen)")
L.append(f"twin rows fraction per fold: {np.round(twin_frac, 3).tolist()}")
L.append("")
for v in VARIANTS:
    s = np.array(scores[v])
    L.append(f"{v}\tmean={s.mean():.4f}\tstd={s.std(ddof=1):.4f}\tfolds={np.round(s, 4).tolist()}")
txt = "\n".join(L)
print("\n" + txt, flush=True)
open("/kaggle/working/cv_report.txt", "w").write(txt + "\n")
'''

CELLS = [("setup", SETUP), ("lora features", LORA), ("cv", CV)]


def build():
    cells = []
    for name, src in CELLS:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True),
        })
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = os.path.join(OUT_DIR, "ntl-cvdiag.ipynb")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    print("written to", path)


if __name__ == "__main__":
    build()
