# Prepare kev-format JSONL for Name That LLM (18-way model attribution).
# Splits by prompt_id (GroupShuffleSplit) so holdout prompts are unseen, mirroring the hidden test set.
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
INSTR = "Which of the 18 candidate LLMs wrote this passage?"
CRITERIA = {m: None for m in MODELS}

def to_record(text, model):
    return {"state": text,
            "questions": {"model": {"type": "choice", "instructions": INSTR,
                                    "criteria": CRITERIA, "label": model, "src": "name_that_llm"}}}

def main():
    df = pd.read_csv("data/train.csv")
    assert set(df["model"]) <= set(MODELS)

    # one grouped 80/20 split for local validation; prompts are never shared across sides
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, ho_idx = next(gss.split(df, df["model"], groups=df["prompt_id"]))
    tr, ho = df.iloc[tr_idx], df.iloc[ho_idx]

    with open("kev_train.jsonl", "w", encoding="utf-8") as f:
        for _, r in tr.iterrows():
            f.write(json.dumps(to_record(r["text"], r["model"]), ensure_ascii=False) + "\n")
    with open("kev_holdout.jsonl", "w", encoding="utf-8") as f:
        for _, r in ho.iterrows():
            f.write(json.dumps(to_record(r["text"], r["model"]), ensure_ascii=False) + "\n")
    with open("kev_full.jsonl", "w", encoding="utf-8") as f:   # final model trains on everything
        for _, r in df.iterrows():
            f.write(json.dumps(to_record(r["text"], r["model"]), ensure_ascii=False) + "\n")

    print(f"train={len(tr)} rows / {tr.prompt_id.nunique()} prompts; "
          f"holdout={len(ho)} rows / {ho.prompt_id.nunique()} prompts")
    print("holdout class counts:", sorted(ho["model"].value_counts().items()))

if __name__ == "__main__":
    main()
