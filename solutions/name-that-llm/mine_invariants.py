"""Mine stylometric invariants on train.csv that are (nearly) exclusive to a few models.

The 0.604 notebook carries one hand-found invariant: ' \u2014 ' (spaced em-dash) appears in
86/2699 training rows and *only* in claude-opus-4-6 / opus-5 / glm-5.3 / glm-5.3-flash, so
any model prediction outside that set is provably wrong and can be hard-corrected.

This script asks whether there are more of that shape. A candidate must be
  - formatting-level (not a content word), so it reflects how a model writes, and
  - pure: every training row carrying it has its label inside a small allowed set.
Reported with coverage, purity and the per-model breakdown so a human can veto.
"""
import itertools
import re
import sys

import numpy as np
import pandas as pd

TRAIN = "../../data/ntl/train.csv"
d = pd.read_csv(TRAIN)
d = d.dropna(subset=["text", "model"])
texts = d["text"].astype(str)
labels = d["model"].astype(str).to_numpy()
N = len(d)
print(f"train rows {N}, models {d['model'].nunique()}, prompts {d['prompt_id'].nunique()}")

# ---------------------------------------------------------------- candidates
CANDIDATES = {}


def add(name, mask):
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n >= 15:
        CANDIDATES[name] = mask


# 1. every rare unicode punctuation / symbol character, individually
chars = {}
for ch in set("".join(texts.tolist())):
    if ord(ch) < 128:
        continue
    m = texts.str.contains(re.escape(ch), regex=True, na=False)
    n = int(m.sum())
    if 15 <= n <= 0.35 * N:
        chars[ch] = m
for ch, m in chars.items():
    add(f"char U+{ord(ch):04X} {ch!r}", m)

# 2. ascii formatting habits
PATTERNS = {
    "double space": r"[^ \n]  +[^ \n]",
    "space before period": r"\w \.",
    "space before comma": r"\w ,",
    "space before colon": r"\w :",
    "no space after period": r"\w\.[A-Za-z]",
    "trailing space eol": r"[ \t]\n",
    "three-space indent": r"\n   [^ \n]",
    "two-space indent": r"\n  [^ \n]",
    "double newline": r"\n\s*\n",
    "ends with newline": r"\n\s*$",
    "markdown bold": r"\*\*[^*\n]+\*\*",
    "markdown italic underscore": r"(?<!\w)_[^_\n]+_(?!\w)",
    "markdown bullet dash": r"(?m)^\s*[-*] ",
    "numbered list": r"(?m)^\s*\d+[.)] ",
    "heading hash": r"(?m)^#{1,6} ",
    "blockquote": r"(?m)^> ",
    "inline code backtick": r"`[^`\n]+`",
    "fenced code": r"```",
    "paren citation": r"\(\d{4}\)",
    "semicolon": r";",
    "exclamation": r"!",
    "question": r"\?",
    "allcaps word": r"\b[A-Z]{4,}\b",
    "colon then newline": r":\s*\n",
    "ellipsis dots": r"\.\.\.",
    "ellipsis char": "\u2026",
    "double hyphen": r"--",
    "arrow ascii": r"->|=>",
    "percent no space": r"\d%",
    "percent with space": r"\d %",
    "numeric range hyphen": r"\d-\d",
}
for name, pat in PATTERNS.items():
    add(name, texts.str.contains(pat, regex=True, na=False))

# 3. first characters of the passage (opening habit)
for k in (2, 3):
    pref = texts.str[:k]
    for val in pref.value_counts().index[:40]:
        if len(val.strip()) < k:
            continue
        add(f"starts with {val!r}", (pref == val).to_numpy())

# 4. per-1000-char density buckets of especially telling characters
for ch in ["\u2014", "\u2013", "(", ")", '"', "'", ";", ":", "\n"]:
    dens = texts.str.count(re.escape(ch)) / texts.str.len().clip(lower=1) * 1000
    for lo, hi in ((0, 0.001), (3, 1e9), (6, 1e9)):
        if hi > 1e6:
            m = (dens >= lo).to_numpy()
        else:
            m = ((dens >= lo) & (dens <= hi)).to_numpy()
        add(f"density {ch!r} in [{lo},{hi if hi < 1e6 else 'inf'})", m)


# ---------------------------------------------------------------- score them
def purity(mask):
    """Return (allowed models, min count of disallowed at 100% rule, coverage)."""
    tab = pd.Series(labels[mask]).value_counts()
    return tab


rows = []
for name, mask in CANDIDATES.items():
    tab = purity(mask)
    k = len(tab)
    if k > 6:
        continue
    cov = mask.mean()
    rows.append((name, int(mask.sum()), cov, k, tab.to_dict()))

rows.sort(key=lambda r: (-r[1]))
print(f"\n{'pattern':44s} {'n':>5s} {'cov':>6s} {'#mdl':>4s}  models (count)")
for name, n, cov, k, tab in rows:
    top = ", ".join(f"{m}:{c}" for m, c in list(tab.items())[:6])
    print(f"{name:44s} {n:5d} {cov*100:5.1f}% {k:4d}  {top}")

# ---------------------------------------------------------------- rule simulation
# For the strongest candidates, estimate how often a *confident* prediction would be
# wrong and how many rows a hard guard could fix. We have no probabilities here, so use
# the label-free proxy: a row's model set must be inside `allowed`; the guard helps only
# if the model would otherwise pick a disallowed label, which we cannot know offline.
# What we CAN report is how much of the training set each rule constrains.
print("\n--- candidate hard guards (purity 100%, >=30 rows) ---")
keep = [(n, c, k, tab) for n, c, cov, k, tab in
        [(r[0], r[1], r[3], r[4]) for r in rows] if k <= 4 and c >= 30]
for n, c, k, tab in keep[:25]:
    print(f"{n:44s} n={c:5d} allowed={list(tab)}")

# ---------------------------------------------------------------- combination check
# A guard is only safe if it never fires on a row whose label is outside `allowed`.
# Verify the shipped em-dash rule end to end and report the residue.
spaced = texts.str.contains(" \u2014 ", regex=False, na=False)
allowed = {"claude-opus-4-6", "claude-opus-5", "glm-5.3", "glm-5.3-flash"}
viol = spaced & ~np.isin(labels, list(allowed))
print(f"\nshipped spaced-em-dash rule: fires on {int(spaced.sum())} rows, "
      f"violations {int(viol.sum())}, coverage {spaced.mean()*100:.2f}%")


if __name__ == "__main__":
    sys.exit(0)
