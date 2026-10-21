"""Side-effect-free stylometry feature extractor (punctuation / markdown / layout statistics)."""
import re

import numpy as np

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
