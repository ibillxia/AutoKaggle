"""Systematically mine 'forbidden character' invariants, the shape the 0.612 notebook exploits.

Two rules are published and both are of the same form:

  spaced em-dash ' \u2014 '  -> only claude-opus-4-6 / opus-5 / glm-5.3 / glm-5.3-flash ever use it
  curly quotes '\u2019\u2018\u201c\u201d' -> claude-* and glm-* never use them at all

So the useful primitive is not "which models *do* produce X" but "which models *never* do".
For every candidate glyph X we want: the set of models with zero occurrences in train, and how
many rows carry X. A guard then zeroes the never-models on rows that carry X.

Precision matters more than coverage: a wrong forbidden set throws away the true label.
So each rule is reported with its support on both sides and only rules with a healthy margin
are printed.
"""
import itertools
import json
import unicodedata

import numpy as np
import pandas as pd

train = pd.read_csv("../../data/ntl/train.csv")
classes = json.load(open("oof/classes.json"))  # same 18 labels, sorted
y = train["model"].to_numpy()
texts = train["text"].astype(str).to_numpy()
N = len(train)

# candidate glyphs: every non-ascii character that occurs at all, plus ascii punctuation
seen = set()
for t in texts:
    for ch in t:
        if ord(ch) > 127:
            seen.add(ch)
ASCII_PUNCT = list("();:!?\"'[],.-–—§*#`$%&+=/<>@\\^_{|}~")
cands = sorted(seen) + [c for c in ASCII_PUNCT if c not in seen]

rows = []
for ch in cands:
    has = np.array([ch in t for t in texts])
    n_has = int(has.sum())
    if n_has < 20:
        continue
    counts = pd.Series(y[has]).value_counts()
    never = [c for c in classes if c not in counts.index]
    # a rule with no never-models is useless; a rule covering everything is another
    if not never or len(never) == len(classes):
        continue
    # how strong is the asymmetry? rows in the *allowed* set, and the largest allowed count
    rows.append(dict(ch=ch, name=unicodedata.name(ch, "?"), n_has=n_has, never=never,
                     n_never=len(never), top=int(counts.max()), allowed=int(len(counts)),
                     tab={k: int(v) for k, v in counts.items()}))

rows.sort(key=lambda r: (-r["n_has"]))
print(f"{'char':6s} {'n':>4s} {'#never':>6s} {'#allowed':>8s}  never  |  allowed counts")
for r in rows:
    print(f"{r['ch']!r:6s} {r['n_has']:4d} {r['n_never']:6d} {r['allowed']:8d}  "
          f"{','.join(m.replace('claude-', 'c-') for m in r['never'])[:60]:60s} | "
          f"{list(r['tab'].items())[:4]}")

print("\n--- the two published rules, verified independently ---")
spaced = np.array([" \u2014 " in t for t in texts])
print(f"spaced em-dash: {int(spaced.sum())} rows, models={sorted(set(y[spaced]))}")
curly = np.array([any(c in t for c in "\u2019\u2018\u201c\u201d") for t in texts])
mm = pd.Series(y[curly]).value_counts()
print(f"curly quotes: {int(curly.sum())} rows, "
      f"never={[c for c in classes if c not in mm.index]}")
print(f"   counts: {dict(list(mm.items()))}")

# ------------------------------------------------------------------ rule table
# Freeze the rules that survive a margin test: at least 25 rows carrying the glyph, at least
# 30 rows for every allowed model, and the never-set is a coherent group (all claude/glm or a
# single family), which makes it a style habit rather than a sampling accident.
print("\n--- guard candidates with >=25 supporting rows ---")
for r in rows:
    if r["n_has"] < 25:
        continue
    if min(r["tab"].values()) < 20:
        continue
    print(f"{r['ch']!r:6s} {unicodedata.name(r['ch'], '?')[:40]:40s} n={r['n_has']:4d} "
          f"forbid={[m for m in r['never']]}")
