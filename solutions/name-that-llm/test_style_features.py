# Does an explicit stylometry block (punctuation / markdown / layout statistics) add anything
# on top of the raw TF-IDF blocks?
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
CFGS = [dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000),
        dict(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=400000),
        dict(analyzer="char_wb", ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_features=500000)]

PUNCT = list(".,;:!?—-–'\"()[]{}<>/\\|@#$%^&*_+=~`")
MARK = ["\n", "- ", "* ", "#", "**", "__", "> ", "1.", "2.", "(", ":", ";", "…", "”", "“", "’", "‘"]

def style_features(texts):
    rows = []
    for t in texts:
        n = max(len(t), 1)
        words = t.split()
        nw = max(len(words), 1)
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        slens = [len(s.split()) for s in sents] or [0]
        f = [n, nw, np.mean([len(w) for w in words]) if words else 0,
             len(sents), np.mean(slens), np.std(slens), len(set(words)) / nw]
        f += [t.count(p) * 100.0 / n for p in PUNCT]
        f += [t.count(m) * 100.0 / n for m in MARK]
        f += [sum(c.isdigit() for c in t) * 100.0 / n, sum(c.isupper() for c in t) * 100.0 / n,
              len(re.findall(r"\n\n", t)), len(re.findall(r"\b\w+\b", t)) / max(len(sents), 1)]
        rows.append(f)
    X = np.asarray(rows, dtype=np.float64)
    return np.log1p(np.abs(X)) * np.sign(X)

df = pd.read_csv("data/train.csv")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, ho = next(gss.split(df, df["model"], groups=df["prompt_id"]))
train, test = df.iloc[tr].reset_index(drop=True), df.iloc[ho].reset_index(drop=True)
truth = test["model"].values

Xtr, Xte = [], []
for cfg in CFGS:
    v = TfidfVectorizer(**cfg).fit(train["text"])
    Xtr.append(v.transform(train["text"])); Xte.append(v.transform(test["text"]))
A, B = hstack(Xtr).tocsr(), hstack(Xte).tocsr()
sc = StandardScaler().fit(style_features(train["text"]))
S_tr, S_te = sc.transform(style_features(train["text"])), sc.transform(style_features(test["text"]))

def run(A, B, tag, C=100):
    clf = LogisticRegression(C=C, max_iter=3000).fit(A, train["model"].values)
    order = [list(clf.classes_).index(m) for m in MODELS]
    P = clf.predict_proba(B)[:, order]
    print(f"{tag:32s} bal_acc={balanced_accuracy_score(truth, [MODELS[i] for i in P.argmax(1)]):.4f}", flush=True)
    return P

run(A, B, "tfidf only")
for w in (0.3, 1.0, 3.0):
    Aw = hstack([A, csr_matrix(S_tr) * w]).tocsr()
    Bw = hstack([B, csr_matrix(S_te) * w]).tocsr()
    run(Aw, Bw, f"tfidf + style block (w={w})")
