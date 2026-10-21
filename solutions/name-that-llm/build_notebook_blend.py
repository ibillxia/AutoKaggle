# Build the blend notebook (v2): raw 3-block TF-IDF + fine-tuned roberta-large, 0.5/0.5 log-prob average.
# No prompt centring: the hidden test set does not share prompts across models, which the first LB run confirmed.
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "ntl_blend.ipynb"
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
    md("""# Name That LLM — TF-IDF + roberta-large blend

Two complementary views of writing style: surface n-grams (word 1-2 + char 2-5 + char 3-6 TF-IDF,
LogReg C=100) and a supervised encoder (roberta-large, 18-way head, full training data).
Log-probabilities are averaged 0.5/0.5 — the peak of a flat plateau on a prompt-disjoint holdout
(TF-IDF 0.307, roberta 0.280, blend 0.350 balanced accuracy)."""),

    code("""\
import glob, os
import numpy as np, pandas as pd
import torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODELS = """ + json.dumps(MODELS) + """

def find(name):
    for pat in (f"/kaggle/input/competitions/name-that-llm/{name}", f"/kaggle/input/name-that-llm/{name}"):
        if os.path.exists(pat):
            return pat
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, name
    return hits[0]

train = pd.read_csv(find("train.csv"))
test = pd.read_csv(find("test.csv"))
print(train.shape, test.shape)"""),

    code("""\
CFGS = [dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000),
        dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000),
        dict(analyzer="char_wb", ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_features=500000)]
Xtr, Xte = [], []
for cfg in CFGS:
    v = TfidfVectorizer(**cfg).fit(train["text"])
    Xtr.append(v.transform(train["text"]))
    Xte.append(v.transform(test["text"]))
A, B = hstack(Xtr).tocsr(), hstack(Xte).tocsr()
print("features:", A.shape, B.shape)

clf = LogisticRegression(C=100, max_iter=3000).fit(A, train["model"].values)
order_tf = [list(clf.classes_).index(m) for m in MODELS]     # predict_proba columns follow clf.classes_
P_tf = clf.predict_proba(B)[:, order_tf]
print("tfidf done", P_tf.shape)"""),

    code("""\
ROB = glob.glob("/kaggle/input/ntl-roberta*")[0]
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained(ROB)
try:
    rob = AutoModelForSequenceClassification.from_pretrained(ROB, dtype=torch.float16).cuda().eval()
except TypeError:                                    # older transformers
    rob = AutoModelForSequenceClassification.from_pretrained(ROB, torch_dtype=torch.float16).cuda().eval()
id2label = {int(k): v for k, v in rob.config.id2label.items()}
assert [id2label[i] for i in range(len(MODELS))] == MODELS, id2label

@torch.no_grad()
def predict_rob(texts, bs=32, maxlen=384):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=maxlen,
                  return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = rob(**enc).logits
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)

P_rob = predict_rob(test["text"].tolist())
print("roberta done", P_rob.shape)"""),

    code("""\
W_ROB = 0.5
P = (1 - W_ROB) * np.log(np.clip(P_tf, 1e-9, 1)) + W_ROB * np.log(np.clip(P_rob, 1e-9, 1))
pred = [MODELS[i] for i in P.argmax(1)]
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
    meta = {"id": "ibillxia/ntl-blend", "title": "ntl blend", "code_file": OUT.name,
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": False,
            "dataset_sources": ["ibillxia/ntl-roberta"], "competition_sources": ["name-that-llm"],
            "kernel_sources": []}
    (HERE / "kernel-metadata-blend.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", OUT)
