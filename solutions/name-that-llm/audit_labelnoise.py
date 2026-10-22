# Label-noise audit for the forum corpora, using the typographic invariants we mined on train.
# A row that carries a glyph its claimed model *never* produces in 150 competition passages is
# either mislabelled or out-of-distribution -- exactly the rows that poison stage-1.
import json

import pandas as pd

PATTERNS = {
    "space_emdash": " \u2014 ",
    "emdash": "\u2014",
    "curly_set": ("\u2019", "\u2018", "\u201c", "\u201d"),
    "straight_apos": "'",
    "straight_dq": '"',
    "curly_dq": ("\u201c", "\u201d"),
    "ellipsis": "\u2026",
    "bullet": "\u2022",
    "nbsp": "\u00a0",
}


def has(text, pat):
    if isinstance(pat, tuple):
        return any(p in text for p in pat)
    return pat in text


def main():
    tr = pd.read_csv("data/train.csv")
    uni = pd.read_parquet("extra/union_all.parquet")
    classes = sorted(tr["model"].unique())

    never = {}
    for name, pat in PATTERNS.items():
        seen = {m for m, t in zip(tr["model"], tr["text"]) if has(t, pat)}
        never[name] = [c for c in classes if c not in seen]
    print("never-sets mined on train (support = rows carrying the glyph):")
    for k, v in never.items():
        support = sum(has(t, PATTERNS[k]) for t in tr["text"])
        print(f"  {k:13s} support={support:4d}  never={len(v):2d}  {','.join(m.replace('claude-','c-') for m in v)[:110]}")

    # a rule is only used when it has a healthy margin: >=3 forbidden models and >=20 support
    keep = {}
    for k, v in never.items():
        support = sum(has(t, PATTERNS[k]) for t in tr["text"])
        if 3 <= len(v) <= 15 and support >= 20:
            keep[k] = v
    print(f"\nrules kept: {list(keep)}")

    def violations(df):
        flag = pd.Series(False, index=df.index)
        for k, models in keep.items():
            bad = df["model"].isin(models) & df["text"].map(lambda t: has(t, PATTERNS[k]))
            flag |= bad
        return flag

    assert violations(tr).sum() == 0, "rule fires on train -- refusing to score noise with it"

    out = []
    for src, g in uni.groupby("source"):
        v = violations(g)
        out.append(dict(source=src, rows=len(g), violate=int(v.sum()),
                        rate=round(100 * v.mean(), 2),
                        mean_words=round(g["text"].str.split().str.len().mean(), 1)))
    df = pd.DataFrame(out).sort_values("rate", ascending=False)
    print("\n" + df.to_string(index=False, justify="left"))
    print(f"\ntrain mean words = {tr['text'].str.split().str.len().mean():.1f}")
    df.to_csv("extra/audit_labelnoise.csv", index=False)


if __name__ == "__main__":
    main()
