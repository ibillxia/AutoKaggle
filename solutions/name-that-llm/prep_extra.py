# Build the extra training corpus (forum post 749629): 24k short responses from the same 18 models.
# Dedupes against the official train texts so CV stays honest.
import hashlib
import re
import pandas as pd

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]

def norm(t):
    return re.sub(r"\s+", " ", str(t)).strip().lower()

def h(t):
    return hashlib.sha256(norm(t).encode()).hexdigest()

def main():
    tr = pd.read_csv("data/train.csv")
    seen = {h(t) for t in tr["text"]}
    print("official train:", len(tr), "unique texts:", len(seen))

    wide = pd.read_parquet("extra/corpus/external_short_wide.parquet")
    long = wide.melt(id_vars=["id", "prompt"], var_name="model", value_name="text").dropna(subset=["text"])
    long = long[long["model"].isin(MODELS)]
    print("melted extra rows:", len(long))

    long["h"] = long["text"].map(h)
    before = len(long)
    long = long.drop_duplicates("h")
    print("dropped intra-corpus dupes:", before - len(long))
    before = len(long)
    long = long[~long["h"].isin(seen)]
    print("dropped rows duplicated in official train:", before - len(long))

    long["wc"] = long["text"].str.split().str.len()
    long = long[(long["wc"] >= 35) & (long["wc"] <= 300)]
    print("after length filter:", len(long))
    print(long["model"].value_counts().to_string())

    long[["id", "prompt", "model", "text"]].to_csv("extra/extra_short.csv", index=False)
    print("wrote extra/extra_short.csv")

if __name__ == "__main__":
    main()
