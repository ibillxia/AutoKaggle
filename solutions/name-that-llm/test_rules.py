"""Evaluate a rules-driven typographic guard on real OOF logits.

The published 0.604/0.612 kernels hard-code two rules found by eye. Mining every glyph
turns up two much wider ones that nobody has published:

  '—'  (em-dash, 1052 rows = 39% of train)  -> never produced by gemini-3.1-pro-preview,
                                               gemini-3.6-flash, gemini-3.8-flash
  "'"  (ascii apostrophe, 835 rows = 31%)   -> never produced by gpt-5.6-sol
  '"'  (ascii double quote, 109 rows)       -> never produced by gemini-3.7-flash,
                                               gpt-5.6-luna/sol/terra, gpt-6-astra, qwen3.8-max

This script measures them one at a time on the OOF matrix, with and without Sinkhorn and the
dense-cluster Hungarian step, so the next kernel carries only rules that pay for themselves.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score

train = pd.read_csv("../../data/ntl/train.csv")
assert hashlib.sha256("\n".join(train["id"]).encode()).hexdigest() == \
    open("oof/train_row_alignment.txt").read().split()[-1]
L = np.load("oof/oof_logits_runB_lucasboesen_lora24_s7.npy")
classes = json.load(open("oof/classes.json"))
y = train["model"].map({c: i for i, c in enumerate(classes)}).to_numpy()
texts = train["text"].astype(str).tolist()
K = 18

# ------------------------------------------------------------------ the rules
# Each rule: if the passage carries any of `chars`, these models are impossible.
RULES = [
    # (name, trigger chars, banned models). The published spaced-em-dash rule is expressed
    # as "only these four may appear", i.e. everything else is banned.
    ("spaced em-dash (published)", [" \u2014 "],
     [c for c in ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
                  "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
                  "gemini-3.8-flash", "claude-opus-4-8", "claude-sonnet-5", "claude-fable-5",
                  "qwen3.8-max", "kimi-k3"]]),
    ("curly quotes (published)", ["\u2019", "\u2018", "\u201c", "\u201d"],
     ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
      "claude-sonnet-5", "glm-5.3", "glm-5.3-flash"]),
    ("em-dash unspaced (NEW)", ["\u2014"],
     ["gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.8-flash"]),
    ("ascii apostrophe (NEW)", ["'"], ["gpt-5.6-sol"]),
    ("ascii double quote (NEW)", ['"'],
     ["gemini-3.7-flash", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
      "gpt-6-astra", "qwen3.8-max"]),
    ("curly double quote", ["\u201c", "\u201d"],
     ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
      "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
      "gemini-3.7-flash", "glm-5.3", "glm-5.3-flash"]),
]


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def ba(P):
    return balanced_accuracy_score(y, P.argmax(1))


def sinkhorn(P, iters=100):
    P = P.copy()
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        P *= target / P.sum(0)
        P /= P.sum(1, keepdims=True)
    return P


def apply_rules(P, rules, verbose=False):
    P = P.copy()
    for name, chars, banned in rules:
        mask_ban = np.array([c in banned for c in classes])
        hit = np.array([any(c in t for c in chars) for t in texts])
        fixed = 0
        for i in np.where(hit)[0]:
            if mask_ban[P[i].argmax()]:
                P[i, mask_ban] = 0.0
                if P[i].sum() <= 0:
                    P[i, ~mask_ban] = 1.0
                P[i] /= P[i].sum()
                fixed += 1
        if verbose:
            outside = (mask_ban[y[hit]]).sum() if hit.any() else 0
            print(f"    {name:32s} hits {int(hit.sum()):5d}  fixed {fixed:4d}  "
                  f"(true label banned: {int(outside)})")
    return P


def hungarian(P, groups, lo=9, hi=18):
    P = P.copy()
    flips = 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if not (lo <= len(idx) <= hi):
            continue
        r, c = linear_sum_assignment(-np.log(np.clip(P[idx], 1e-12, 1.0)))
        for a, b in zip(r, c):
            if P[idx[a]].argmax() != b:
                flips += 1
        for a, b in zip(r, c):
            P[idx[a], :] = 0.0
            P[idx[a], b] = 1.0
    return P, flips


P0 = softmax(L)
print(f"baseline ba={ba(P0):.4f}\n")

print("-- cumulative rules on raw softmax --")
rules = []
for r in RULES:
    rules.append(r)
    P = apply_rules(P0, rules)
    tag = " +".join(x[0].split(" (")[0] for x in rules)
    print(f"  ba={ba(P):.4f} ({ba(P)-ba(P0):+.4f})  <= {tag}")

print("\n-- rule-by-rule ablation (each single rule alone) --")
for r in RULES:
    P = apply_rules(P0, [r])
    print(f"  ba={ba(P):.4f} ({ba(P)-ba(P0):+.4f})  {r[0]}")

print("\n-- with sinkhorn first --")
P_sk = sinkhorn(P0)
print(f"  sinkhorn alone ba={ba(P_sk):.4f} ({ba(P_sk)-ba(P0):+.4f})")
P_sk_all = apply_rules(P_sk, RULES)
print(f"  sinkhorn + all rules ba={ba(P_sk_all):.4f} ({ba(P_sk_all)-ba(P0):+.4f})")

print("\n-- then the dense-cluster hungarian on top (tfidf avg@0.88) --")
X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
D = np.clip(1 - (X @ X.T).toarray(), 0, None)
np.fill_diagonal(D, 0)
groups = fcluster(linkage(squareform(D, checks=False), method="average"), 0.88, criterion="distance")
for name, base in (("raw", P0), ("rules only", apply_rules(P0, RULES)),
                   ("sinkhorn+rules", P_sk_all)):
    Ph, fl = hungarian(base, groups, lo=9, hi=18)
    print(f"  {name:16s} -> ba={ba(Ph):.4f} ({ba(Ph)-ba(P0):+.4f}) flips={fl}")

print("\n-- how often does each rule ban the TRUE label? (must be 0 or near 0) --")
for r in RULES:
    name, chars, banned = r
    m = np.array([c in banned for c in classes])
    hit = np.array([any(c in t for c in chars) for t in texts])
    bad = int((m[y] & hit).sum())
    print(f"  {name:32s} hits {int(hit.sum()):5d}  true-label-banned {bad}")
