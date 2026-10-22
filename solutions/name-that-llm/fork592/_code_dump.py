# ==================== cell 1 ====================

import gc, glob, json, os, re, time
import numpy as np
import pandas as pd
import torch
from torch import nn

T0 = time.time()
INPUT_DIR = "/kaggle/input"

# Blend weights, fit on out-of-fold predictions (log probabilities are multiplied by these and summed).
W_SVC = 1.021
LORAS = {  # checkpoint folder: blend weight, centred-twin weight, batch size, base model to look for
    "lora8b_v1":  dict(w=0.226, twin_w=0.079, bs=16, base=dict(model_type="qwen3", hidden_size=4096, num_hidden_layers=36)),
    "lora14b_v1": dict(w=0.429, twin_w=0.150, bs=8,  base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),
}
TWIN = dict(thresh=0.85, min_size=2, max_size=18)   # topic clusters that get the centred twin

# Column order of the checkpoints' heads (checked against each checkpoint's meta.json below).
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
K = len(LABELS)


def find(pattern):
    hits = sorted(glob.glob(f"{INPUT_DIR}/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


train = pd.read_csv(find("train.csv"))
test = pd.read_csv(find("test.csv"))
test["text"] = test["text"].fillna("").astype(str)
y = train["model"].map({l: i for i, l in enumerate(LABELS)}).values
LOGIT = np.zeros((len(test), K))   # every member adds w * log p here
print("train", train.shape, "test", test.shape, flush=True)


# ==================== cell 3 ====================

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


pool = pd.read_parquet(find("pool_v1.parquet"))
fit_texts = train.text.astype(str).tolist() + pool.text.tolist()
fit_y = np.r_[y, pool.model.map({l: i for i, l in enumerate(LABELS)}).values]
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
    B.append(vec.transform([prep(t) for t in test.text]) * weight)
svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced").fit(hstack(A, format="csr"), fit_y)
P_svc = softmax(svc.decision_function(hstack(B, format="csr")))
LOGIT += W_SVC * np.log(np.clip(P_svc, 1e-9, 1))
del A, B, svc, pool
gc.collect()
print(f"n-gram model done ({time.time() - T0:.0f}s)", flush=True)


# ==================== cell 5 ====================

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
    """First `keep` decoder layers, split over both T4s when it does not fit on one."""
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
    """Pooled features Z (n, 8 * hidden) and the checkpoint's heads."""
    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
    assert meta["labels"] == LABELS, "label order differs from the checkpoint's heads"
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
    order = np.argsort([len(x) for x in ids], kind="stable")   # similar lengths together = less padding
    Z = np.zeros((len(ids), H * len(layers)), np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(model.embed_tokens.weight.device)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        if not all(torch.isfinite(hs[l]).all() for l in layers):   # fp16 overflow: redo this batch in bf16
            model.to(torch.bfloat16)
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            model.to(dtype)
        m = enc["attention_mask"].unsqueeze(-1).float().to("cuda:0")
        m[:, 0] = 0   # the first token is an attention sink; leave it out of the mean
        den = m.sum(1).clamp(min=1)
        Z[idx] = torch.cat([nm((hs[l].to("cuda:0").float() * m).sum(1) / den) for l, nm in zip(layers, norms)], -1).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return Z, {k: v.float().numpy() for k, v in heads.items() if not k.startswith("norms.")}


# ==================== cell 7 ====================

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(test.text.tolist())
D = np.clip(1 - (Xt @ Xt.T).toarray(), 0, None)
np.fill_diagonal(D, 0)
groups = fcluster(linkage(squareform(D, checks=False), method="average"), TWIN["thresh"], criterion="distance") \
    if len(test) >= 2 else np.ones(len(test), dtype=int)
size = np.bincount(groups)[groups]
twin_rows = (size >= TWIN["min_size"]) & (size <= TWIN["max_size"])
del D
print(f"topic clusters: {int((size == 1).sum())} singletons, {int(twin_rows.sum())} passages in clusters of 2-18", flush=True)

for ckpt_name, mc in LORAS.items():
    ckpt = os.path.dirname(find(f"{ckpt_name}/meta.json"))
    Z, heads = lora_features(ckpt, find_base(mc["base"]), test.text.tolist(), mc["bs"])
    LOGIT += mc["w"] * np.log(np.clip(softmax(Z @ heads["plain.weight"].T + heads["plain.bias"]), 1e-9, 1))
    if twin_rows.any():
        Zc = Z.copy()
        for g in np.unique(groups[twin_rows]):
            Zc[groups == g] -= Zc[groups == g].mean(0)
        centred = Zc[twin_rows] @ heads["centered.weight"].T + heads["centered.bias"]
        LOGIT[twin_rows] += mc["twin_w"] * np.log(np.clip(softmax(centred), 1e-9, 1))
    del Z
    gc.collect()
    print(f"{ckpt_name} done ({time.time() - T0:.0f}s)", flush=True)


# ==================== cell 9 ====================

P = softmax(LOGIT)
if len(test) >= 50:   # skip on the 3-row public test.csv
    target = np.full(K, len(P) / K)
    for _ in range(100):
        P *= target / P.sum(0)
        P /= P.sum(1, keepdims=True)
sub = pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[P.argmax(1)]})
sub.to_parquet("submission.parquet", index=False)
print(sub["model"].value_counts().to_string())
print(f"done in {time.time() - T0:.0f}s")
