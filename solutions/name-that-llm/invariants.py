# Typographic invariants mined on the official train: for each glyph, the models that never emit it.
# Used both as a guard (post-processing) and as a label-noise detector (data auditing).
import pandas as pd

PATTERNS = {
    "space_emdash": " \u2014 ",
    "emdash": "\u2014",
    "curly_set": ("\u2019", "\u2018", "\u201c", "\u201d"),
    "straight_apos": "'",
    "straight_dq": '"',
    "curly_dq": ("\u201c", "\u201d"),
}
MIN_SUPPORT = 20
MIN_NEVER, MAX_NEVER = 3, 15


def has(text, pat):
    if isinstance(pat, tuple):
        return any(p in text for p in pat)
    return pat in text


def mine(train):
    """Return {pattern_name: [models that never use it]} keeping only rules with a real margin."""
    classes = sorted(train["model"].unique())
    rules = {}
    for name, pat in PATTERNS.items():
        support = sum(has(t, pat) for t in train["text"])
        if support < MIN_SUPPORT:
            continue
        seen = {m for m, t in zip(train["model"], train["text"]) if has(t, pat)}
        never = [c for c in classes if c not in seen]
        if MIN_NEVER <= len(never) <= MAX_NEVER:
            rules[name] = never
    return rules


def violations(df, rules, text_col="text", label_col="model"):
    flag = pd.Series(False, index=df.index)
    for name, models in rules.items():
        pat = PATTERNS[name]
        flag |= df[label_col].isin(models) & df[text_col].map(lambda t: has(t, pat))
    return flag


def rules_from_train(path="data/train.csv"):
    tr = pd.read_csv(path)
    r = mine(tr)
    assert not violations(tr, r).any(), "invariant fires on train"
    return r
