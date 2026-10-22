"""Build ibillxia/ntl-hybrid2: hybrid + a wider, mined typographic guard.

hybrid (v1) ships the guard published in the 0.604 notebook: one hand-found em-dash rule and a
Hungarian pass over topic clusters of 12-18 at tfidf average-linkage 0.85.

Measuring that machinery on real OOF logits (see test_rules.py / test_grouping.py) showed two
things it leaves on the table:

  1. the guard's size is set almost entirely by grouping quality -- the same Hungarian step is
     worth +0.150 at threshold 0.85 and +0.177 at 0.90 (pairwise F1 0.877 -> 0.955);
  2. four more "this model never produces that glyph" invariants exist and each has zero
     true-label contradictions over its hits, adding +0.004 on top of the two published ones.

So v2 keeps the model blend of v1 byte-for-byte and changes only the post-processing:
  - six typographic rules instead of one (spaced em-dash, curly quotes, em-dash, ascii
    apostrophe, ascii double quote, curly double quote), applied post-Sinkhorn;
  - the Hungarian pass uses its own 0.90 clustering (the twin keeps its published 0.85) and
    covers every cluster of 2-18 rows, not just 12-18;
  - the assignment cost makes a ruled-out label prohibitive rather than merely unlikely, so
    the matching cannot undo a rule.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_NB = os.path.join(HERE, "fork604b", "ntl-fork604b.ipynb")
OUT_DIR = os.path.join(HERE, "hybrid2")
DST_NB = os.path.join(OUT_DIR, "ntl-hybrid2.ipynb")

FOLD_W, TWIN_RATIO = 0.07, 0.35
FOLDS = ["ts14b_f%d" % i for i in range(5)]

OLD_LORAS = '''    "lora14b_v1": dict(w=0.429, twin_w=0.150, bs=8,  base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),
}'''

NEW_LORAS_HEAD = '''    "lora14b_v1": dict(w=0.429, twin_w=0.150, bs=8,  base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),
    # --- the five two-stage 14B folds of zhuyu2/name-that-llm-ts-lora: the 0.576 -> 0.601 gain.
'''

GUARD_START = "# --- DATA-VALIDATED UPGRADE 1: POST-SINKHORN SPACED EM-DASH GUARD ---"
GUARD_END = "pred_indices = P.argmax(1)"

NEW_GUARD = '''# --- TYPOGRAPHIC INVARIANTS, MINED FROM train.csv ---------------------------------------
# Each rule reads "if the passage carries any of these glyphs, the listed models cannot have
# written it". Style habits, not content: every rule was verified over every training row and
# has ZERO true-label contradictions, so a hard filter can only remove impossible answers.
# hits = training rows the rule fires on (of 2699), 0.604 published the first two.
_RULES = [
    ("spaced em-dash", [" \\u2014 "], 86,
     [l for l in ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
                  "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
                  "gemini-3.8-flash", "claude-opus-4-8", "claude-sonnet-5", "claude-fable-5",
                  "qwen3.8-max", "kimi-k3"]]),
    ("curly quotes", ["\\u2019", "\\u2018", "\\u201c", "\\u201d"], 607,
     ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
      "claude-sonnet-5", "glm-5.3", "glm-5.3-flash"]),
    ("em-dash", ["\\u2014"], 1052,
     ["gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.8-flash"]),
    ("ascii apostrophe", ["'"], 835, ["gpt-5.6-sol"]),
    ("ascii double quote", ['"'], 109,
     ["gemini-3.7-flash", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
      "gpt-6-astra", "qwen3.8-max"]),
    ("curly double quote", ["\\u201c", "\\u201d"], 93,
     ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
      "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
      "gemini-3.7-flash", "glm-5.3", "glm-5.3-flash"]),
]

_texts = test.text.astype(str).tolist()
for _name, _chars, _hits, _banned in _RULES:
    _ban = np.array([l in _banned for l in LABELS])
    _fire = np.array([any(c in t for c in _chars) for t in _texts])
    _fixed = 0
    for _i in np.where(_fire)[0]:
        if _ban[P[_i].argmax()]:
            P[_i, _ban] = 0.0
            if P[_i].sum() <= 0:          # Sinkhorn left nothing legal: fall back to the 4 allowed
                P[_i, ~_ban] = 1.0
            P[_i] /= P[_i].sum()
            _fixed += 1
    if _fixed:
        print(f"  rule {_name:20s} fired on {int(_fire.sum()):5d} rows, corrected {_fixed}", flush=True)

# --- DENSE PROMPT BIPARTITE MATCHING -----------------------------------------------------
# One prompt holds at most one passage per model (train: 0 duplicates in 150 prompts), so a
# recovered prompt is a bipartite matching problem, not 18 independent argmaxes.
# The twin's grouping stays at the published 0.85; the matching wants the tighter 0.90 cut,
# which recovers prompts far better (pairwise F1 0.877 -> 0.955 on train) and only ever splits
# rather than merges, so it cannot force distinct labels across two different prompts.
try:
    from scipy.optimize import linear_sum_assignment
    _xg = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(_texts)
    _dg = np.clip(1 - (_xg @ _xg.T).toarray(), 0, None)
    np.fill_diagonal(_dg, 0)
    _guard_groups = (fcluster(linkage(squareform(_dg, checks=False), method="average"), 0.90,
                              criterion="distance") if len(_texts) >= 2
                     else np.ones(len(_texts), dtype=int))
    del _dg, _xg
    _hung = 0
    _clusters = 0
    for _gid in np.unique(_guard_groups):
        _m = np.where(_guard_groups == _gid)[0]
        if not (2 <= len(_m) <= 18):
            continue
        _clusters += 1
        _cost = -np.log(np.clip(P[_m, :], 1e-12, 1.0))
        _cost = np.where(P[_m, :] <= 1e-12, 1e6, _cost)   # a ruled-out label must never be matched
        _r, _c = linear_sum_assignment(_cost)
        for _a, _b in zip(_r, _c):
            if P[_m[_a]].argmax() != _b:
                _hung += 1
        for _a, _b in zip(_r, _c):
            P[_m[_a], :] = 0.0
            P[_m[_a], _b] = 1.0
    print(f"bipartite matching resolved {_hung} collisions across {_clusters} prompts", flush=True)
except Exception as _e:
    print(f"bipartite matching skipped: {_e}", flush=True)

'''

EXTRA_DATASETS = ["zhuyu2/name-that-llm-ts-lora"]


def main():
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    assert len(nb["cells"]) == 1
    src = nb["cells"][0]["source"]
    assert src.count(OLD_LORAS) == 1, "LORAS dict changed upstream"
    entries = "".join(
        '    "%s": dict(w=%.3f, twin_w=%.4f, bs=8, base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),\n'
        % (n, FOLD_W, FOLD_W * TWIN_RATIO) for n in FOLDS)
    src = src.replace(OLD_LORAS, NEW_LORAS_HEAD + entries + "}", 1)

    i = src.index(GUARD_START)
    j = src.index(GUARD_END)
    assert i < j, "guard block not found in upstream order"
    src = src[:i] + NEW_GUARD + src[j:]
    compile(src, "<hybrid2>", "exec")
    nb["cells"][0]["source"] = src

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(nb, open(DST_NB, "w", encoding="utf-8"), indent=1)

    meta = json.load(open(os.path.join(HERE, "fork604b", "kernel-metadata.json"), encoding="utf-8"))
    meta["id"] = "ibillxia/ntl-hybrid2"
    meta["title"] = "ntl hybrid2 mined guard"
    meta["code_file"] = "ntl-hybrid2.ipynb"
    meta["is_private"] = True
    for d in EXTRA_DATASETS:
        if d not in meta["dataset_sources"]:
            meta["dataset_sources"].append(d)
    json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    print("wrote", DST_NB)
    print("datasets", len(meta["dataset_sources"]), "kernels", len(meta["kernel_sources"]),
          "models", len(meta["model_sources"]))


if __name__ == "__main__":
    main()
