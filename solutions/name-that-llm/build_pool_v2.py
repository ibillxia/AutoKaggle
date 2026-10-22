# Build the stage-1 corpus we want the next training kernel to use:
# pool_v1 (what the 0.59x recipe trains on) + the marginal rows that passed the audit.
import argparse
import hashlib
import json
import os
import re

import pandas as pd

from invariants import rules_from_train, violations


def norm(t):
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def h(t):
    return hashlib.sha256(norm(t).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="oleg_corpus,maniceet,dhruv_clean,lucas_proxy,prvsiyan_ext,abdullah",
                    help="comma list of marginal sources to add; '' means all")
    ap.add_argument("--clean", action="store_true", help="drop invariant-violating rows")
    ap.add_argument("--per-model", type=int, default=0, help="cap rows per model (0 = no cap)")
    ap.add_argument("--out", default="extra/pool_v2.parquet")
    a = ap.parse_args()

    uni = pd.read_parquet("extra/union_all.parquet")
    tr = pd.read_csv("data/train.csv")
    pool = uni[uni.in_pool_v1].copy()
    marg = uni[(~uni.in_train) & (~uni.in_pool_v1)].copy()
    if a.sources:
        keep = a.sources.split(",")
        marg = marg[marg.source.isin(keep)]
    if a.clean:
        rules = rules_from_train()
        pool = pool[~violations(pool, rules)]
        marg = marg[~violations(marg, rules)]

    both = pd.concat([pool, marg], ignore_index=True)
    both["prompt_id"] = both.get("prompt_key", "")
    out = both[["prompt_id", "model", "text", "source"]]
    out["h"] = out["text"].map(h)
    out = out.drop_duplicates("h").drop(columns="h")

    if a.per_model:
        out = (out.sample(frac=1.0, random_state=0)
                  .groupby("model", sort=False).head(a.per_model)
                  .reset_index(drop=True))

    # keep the official train texts out no matter what
    seen = {h(t) for t in tr["text"]}
    out = out[~out["text"].map(h).isin(seen)].reset_index(drop=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_parquet(a.out, index=False)
    stats = dict(rows=len(out), clean=bool(a.clean), sources=a.sources,
                 per_model=out["model"].value_counts().to_dict(),
                 by_source=out["source"].value_counts().to_dict())
    print(json.dumps({k: v for k, v in stats.items() if k != "per_model"}, indent=1))
    print(out["model"].value_counts().to_string())
    json.dump(stats, open(os.path.splitext(a.out)[0] + "_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
