# Build the final submission notebook: TF-IDF/style + Qwen3.5-4B early-layer embeddings
# + k=3 style-cluster prior (geo reweighting). Validated on 5 grouped holdouts.
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "ntl_final.ipynb"
MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
G3 = {}
for _m in MODELS[:5] + ["qwen3.8-max", "kimi-k3"]:
    G3[_m] = "gptish"
for _m in MODELS[5:9]:
    G3[_m] = "gemini"
for _m in MODELS[9:14] + ["glm-5.3", "glm-5.3-flash"]:
    G3[_m] = "claudeish"
GROUPS = ["gemini", "gptish", "claudeish"]
W_EMB = 0.6      # blend weight of the 4B embeddings
ALPHA = 0.6      # strength of the group prior


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = [
    md(f"""# Name That LLM — TF-IDF/style + Qwen3.5-4B early layers + style-cluster prior

Three components, all validated on prompt-disjoint holdouts:

1. **surface n-grams** — word 1-2 + char 2-5 + char 3-6 TF-IDF with a stylometry block, LogReg C=100
2. **Qwen3.5-4B early layers** — mean-pooled layers 1/3/6/9/16/24/32, LogReg
3. **3 style-cluster prior** — `log P + {ALPHA} * log P_group`, groups are
   `gemini family | gpt + qwen + kimi | claude + glm`

5-seed mean balanced accuracy: TF-IDF 0.346, 4B 0.417, blend {W_EMB:.1f}/{1 - W_EMB:.1f} 0.441,
+ prior **0.446**. No prompt-based tricks: the hidden test set does not share prompts across models.
"""),

    code("""\
import glob, os, time
import numpy as np, pandas as pd
import torch
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODELS = """ + json.dumps(MODELS) + """
G3 = """ + json.dumps(G3) + """
GROUPS = """ + json.dumps(GROUPS) + """

def find(name):
    for pat in (f"/kaggle/input/competitions/name-that-llm/{name}", f"/kaggle/input/name-that-llm/{name}"):
        if os.path.exists(pat):
            return pat
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, name
    return hits[0]

train = pd.read_csv(find("train.csv"))
test = pd.read_csv(find("test.csv"))
print(train.shape, test.shape, train["model"].nunique(), "classes")"""),

    code((HERE / "style_feat.py").read_text(encoding="utf-8") + """

CFGS = [dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000),
        dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000),
        dict(analyzer="char_wb", ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_features=500000)]
Xtr, Xte = [], []
for cfg in CFGS:
    v = TfidfVectorizer(**cfg).fit(train["text"])
    Xtr.append(v.transform(train["text"]))
    Xte.append(v.transform(test["text"]))
sc_s = StandardScaler().fit(style_features(train["text"]))
A = hstack(Xtr + [csr_matrix(sc_s.transform(style_features(train["text"])) * 0.3)]).tocsr()
B = hstack(Xte + [csr_matrix(sc_s.transform(style_features(test["text"])) * 0.3)]).tocsr()
print("tfidf features:", A.shape, B.shape)

clf_tf = LogisticRegression(C=100, max_iter=3000).fit(A, train["model"].values)
P_tf = clf_tf.predict_proba(B)[:, [list(clf_tf.classes_).index(m) for m in MODELS]]

clf_g = LogisticRegression(C=100, max_iter=3000, class_weight="balanced").fit(
    A, np.array([G3[m] for m in train["model"].values]))
P_g = clf_g.predict_proba(B)[:, [list(clf_g.classes_).index(g) for g in GROUPS]]
print("tfidf + group models done")"""),

    code("""\
# ---------- Qwen3.5-4B early-layer embeddings ----------
def find_dir(*names):
    for pat in ("/kaggle/input/{}", "/kaggle/input/datasets/ibillxia/{}", "/kaggle/input/datasets/*/{}"):
        for n in names:
            for hit in glob.glob(pat.format(n)):
                if os.path.isdir(hit):
                    return hit
    for n in names:
        hits = [h for h in glob.glob(f"/kaggle/input/**/{n}", recursive=True) if os.path.isdir(h)]
        if hits:
            return hits[0]
    raise FileNotFoundError(names)

DS = find_dir("ntl-qwen4b", "ntl-qwen4b*")
print("model dir:", DS, sorted(os.listdir(DS))[:5])
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(DS)
LAYERS = [1, 3, 6, 9, 16, 24, 32]
MAXLEN = 320
HIDDEN = None

def load_body(dtype):
    kw = dict(device_map={"": 0}) if dtype is not torch.float32 else dict(device_map="auto")
    try:
        return AutoModel.from_pretrained(DS, dtype=dtype, **kw).eval()
    except TypeError:                                    # transformers < 4.56
        return AutoModel.from_pretrained(DS, torch_dtype=dtype, **kw).eval()

@torch.no_grad()
def embed(texts, batch=8):
    order = np.argsort([len(t.split()) for t in texts])
    out = np.zeros((len(texts), len(LAYERS) * HIDDEN), dtype=np.float16)
    t0 = time.time()
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        enc = tokenizer([texts[j] for j in idx], padding=True, truncation=True,
                        max_length=MAXLEN, return_tensors="pt").to("cuda:0")
        hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
        parts = []
        for L in LAYERS:
            x = hs[L].float()
            if x.device != mask.device:                  # device_map="auto" may shard layers
                x = x.to(mask.device)
            parts.append(((x * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu())
        out[idx] = torch.cat(parts, 1).to(torch.float16).numpy()
        if i % (batch * 50) == 0:
            el = time.time() - t0
            print(f"  {i + len(idx)}/{len(texts)} {el / max(i + len(idx), 1) * 1000:.0f} ms/text", flush=True)
    return out

# T4 lacks native bf16 and Qwen's hybrid blocks can overflow fp16: probe the first rows, then fall back.
for dt in (torch.float16, torch.bfloat16, torch.float32):
    model = load_body(dt)
    HIDDEN = model.config.text_config.hidden_size
    probe = embed(test["text"].tolist()[:min(8, len(test))], batch=4)
    if not np.isnan(probe).any():
        print("using dtype", dt, "hidden", HIDDEN)
        break
    print("dtype", dt, "-> NaN, falling back")
    del model
    torch.cuda.empty_cache()

E_test = embed(test["text"].tolist())
assert not np.isnan(E_test).any()
print("test embeddings:", E_test.shape)

d = np.load(f"{DS}/emb_qwen4b_train.npz", allow_pickle=True)
E_train = np.concatenate([d[f"L{L}"] for L in LAYERS], 1)
assert (d["y"] == train["model"].values).all(), "train embedding order mismatch"
print("train embeddings:", E_train.shape)"""),

    code("""\
# ---------- 4B classifier + blend ----------
sc_e = StandardScaler().fit(E_train)
clf_e = LogisticRegression(C=1.0, max_iter=2000).fit(sc_e.transform(E_train).astype(np.float32),
                                                     train["model"].values)
P_e = clf_e.predict_proba(sc_e.transform(E_test).astype(np.float32))[
    :, [list(clf_e.classes_).index(m) for m in MODELS]]

W_EMB, ALPHA = """ + f"{W_EMB}, {ALPHA}" + """
GOF = np.array([GROUPS.index(G3[m]) for m in MODELS])
L = (1 - W_EMB) * np.log(np.clip(P_tf, 1e-9, 1)) + W_EMB * np.log(np.clip(P_e, 1e-9, 1))
L = L + ALPHA * np.log(np.clip(P_g, 1e-9, 1))[:, GOF]
pred = [MODELS[i] for i in L.argmax(1)]
print(pd.Series(pred).value_counts().to_string())

sub = pd.DataFrame({"id": test["id"], "model": pred})
assert len(sub) == len(test) and sub["id"].is_unique and sub["model"].notna().all()
assert set(sub["model"]) <= set(MODELS)
sub.to_parquet("submission.parquet", index=False)
print("wrote submission.parquet", sub.shape)"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}

if __name__ == "__main__":
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": "ibillxia/ntl-final", "title": "ntl final", "code_file": OUT.name,
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": False,
            "dataset_sources": ["ibillxia/ntl-qwen4b"],
            "competition_sources": ["name-that-llm"], "kernel_sources": []}
    (HERE / "kernel-metadata-final.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", OUT)
