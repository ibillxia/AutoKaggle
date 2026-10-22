# Merge every forum corpus we downloaded into one deduped long table, and measure
# how much of it is genuinely new relative to (a) official train, (b) the public
# in-style pool that the current LoRA recipe already trains on.
import hashlib
import os
import re

import pandas as pd

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]

D = "extra/ds"
# (source tag, path, label col, prompt col, weight col hint)
SOURCES = [
    ("prvsiyan_ext",  f"{D}/prvsiyan__prvsiyans-extended-name-that-llm-dataset/passages_clean.csv", "model", "prompt_id"),
    ("seyit_claude",  f"{D}/seyitkaangunes__ntl-extra-api-passages/ntl_extra_passages_claude.csv", "model", "prompt_id"),
    ("willrice",      f"{D}/willrice__name-that-llm-openrouter-harvest/passages.csv", "label", "prompt_id"),
    ("zbigniew_gpt",  f"{D}/zbigniewkirchner__ntl-extra-gpt-passages/ntl_gpt_passages.csv", "model", "prompt_id"),
    ("maniceet",      f"{D}/maniceet__name-that-llm-openrouter-openai-generations/ntl_openrouter_openai_generations.parquet", "model", "prompt_id"),
    ("lucas_proxy",   f"{D}/lucasboesen__ntl-free-proxy-generations/gen_passages.csv", "model", "prompt_id"),
    ("abdullah",      f"{D}/abdullah1afify__ntl-generations/ntl_generations.csv", "label", "prompt_id"),
    ("dophuchung",    f"{D}/dophuchung__ntl-proxy-generations/ntl_proxy_generations.csv", "label", "prompt_id"),
    ("prvsiyan_own",  f"{D}/prvsiyan__ntl-own-prompt-generations-v1/generated_v1.csv", "label", "prompt_id"),
    ("michael",       f"{D}/michaelsemenoff__ntl-own-extra-v1/ntl_own_extra_v1.csv", "model", None),
    ("hailey",        f"{D}/haileycheng__name-that-llm-same-prompt-passages/passages.csv", "label", "prompt_id"),
    ("dhruv_clean",   f"{D}/dhruvtibarewal__nlm-matched-clean/nlm_matched_clean.parquet", "model", None),
    ("dhruv_pool",    f"{D}/dhruvtibarewal__nlm-matched-clean/nlm_pool_v2.parquet", "model", "prompt_key"),
    ("alperen",       f"{D}/alperen5252525_ntl-vertex-gemini-proxy-gpt-generations/ntl_alp_generations.csv", "label", "prompt_id"),
    ("lxdlxdlxd",     f"{D}/lxdlxdlxd_ntl-synth-generations-v1/generations.parquet", "label", "pid"),
    ("pool_v1",       f"{D}/prvsiyan_ntl-public-in-style-pool-v1/pool_v1.parquet", "model", "prompt_id"),
    ("oleg_corpus",   "extra/extra_short.csv", "model", "prompt"),
]


def norm(t):
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def h(t):
    return hashlib.sha256(norm(t).encode()).hexdigest()


def main():
    tr = pd.read_csv("data/train.csv")
    train_h = {h(t): m for t, m in zip(tr["text"], tr["model"])}
    print(f"official train {len(tr)} rows, {len(train_h)} unique texts")

    frames, stats = [], []
    for tag, path, lab, pcol in SOURCES:
        if not os.path.exists(path):
            print(f"!! missing {tag}: {path}")
            continue
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)
        n0 = len(df)
        sub = pd.DataFrame({
            "text": df["text"].astype(str),
            "model": df[lab].astype(str).str.strip(),
            "prompt_key": df[pcol].astype(str) if pcol and pcol in df.columns else "",
            "source": tag,
        })
        sub = sub[sub["model"].isin(MODELS)]
        n_lab = len(sub)
        wc = sub["text"].str.split().str.len()
        sub = sub[(wc >= 35) & (wc <= 300)].copy()
        n_len = len(sub)
        sub["h"] = sub["text"].map(h)
        n_dup_intra = int(sub.duplicated("h").sum())
        sub = sub.drop_duplicates("h")
        sub["in_train"] = sub["h"].isin(train_h)
        frames.append(sub)
        stats.append(dict(source=tag, raw=n0, mapped=n_lab, len_ok=n_len,
                          dup_intra=n_dup_intra, unique=len(sub),
                          in_train=int(sub["in_train"].sum())))
        print(f"{tag:14s} raw={n0:6d} mapped={n_lab:6d} len_ok={n_len:6d} "
              f"uniq={len(sub):6d} in_train={int(sub['in_train'].sum()):5d}")

    allrows = pd.concat(frames, ignore_index=True)
    pool_addrs = set(allrows.loc[allrows.source == "pool_v1", "h"])
    allrows["in_pool_v1"] = allrows["h"].isin(pool_addrs)

    # one row per unique text, keeping the most trustworthy source order (pool first)
    order = [s[0] for s in SOURCES]
    allrows["src_rank"] = allrows["source"].map({s: i for i, s in enumerate(order)})
    uniq = (allrows.sort_values("src_rank").drop_duplicates("h").reset_index(drop=True))

    print(f"\nunion unique texts: {len(uniq)}")
    print(f"  in official train : {int(uniq['in_train'].sum())}")
    print(f"  in pool_v1        : {int((~uniq['in_train'] & uniq['in_pool_v1']).sum())}")
    marg = uniq[~uniq["in_train"] & ~uniq["in_pool_v1"]]
    print(f"  MARGINAL (new vs both): {len(marg)}")
    print("\nmarginal rows per model:")
    print(marg["model"].value_counts().reindex(MODELS).to_string())
    print("\nmarginal rows per source-of-record:")
    print(marg["source"].value_counts().to_string())

    pd.DataFrame(stats).to_csv("extra/audit_sources.csv", index=False)
    uniq.to_parquet("extra/union_all.parquet", index=False)
    marg.to_parquet("extra/union_marginal.parquet", index=False)
    print("\nwrote extra/union_all.parquet / extra/union_marginal.parquet / extra/audit_sources.csv")


if __name__ == "__main__":
    main()
