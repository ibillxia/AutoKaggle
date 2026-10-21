# Build the offline Kaggle notebook for Name That LLM (kev inference -> submission.parquet).
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "ntl_submission.ipynb"

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

cells = [
    md("# Name That LLM — kev-0.8B fine-tuned attribution\nOffline inference over the hidden test set, writes `submission.parquet`."),

    code("""\
import os, sys, glob, shutil, torch
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

RUN_IN = glob.glob("/kaggle/input/ntl-kev-run")[0]
WORK = "/kaggle/working/kev-run"
shutil.copytree(RUN_IN, WORK, dirs_exist_ok=True, ignore=shutil.ignore_patterns("kev_src"))
sys.path.insert(0, os.path.join(RUN_IN, "kev_src"))

BASE = glob.glob("/kaggle/input/ntl-qwen-base")[0]
# patch the checkpoint's base reference to the dataset mount
d = torch.load(f"{WORK}/head.pt", map_location="cpu")
print("old base:", d.get("base"), "| revision:", d.get("base_revision"))
d["base"], d["base_revision"] = BASE, None
torch.save(d, f"{WORK}/head.pt")
print("patched base ->", BASE)"""),

    code("""\
import pandas as pd
from kev.checkpoint import LoadOptions
from kev.predictors import LocalPredictor
from kev.suite import SERVING_CONTEXT

test = pd.read_csv("/kaggle/input/competitions/name-that-llm/test.csv")
print(test.shape, test.columns.tolist())

opts = LoadOptions(dtype=torch.bfloat16, backend="torch", merge=True)
predictor = LocalPredictor(WORK, "cuda", opts, context=SERVING_CONTEXT)
print("loaded, T =", predictor.checkpoint.meta.temperature)"""),

    code("""\
MODELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
INSTR = "Which of the 18 candidate LLMs wrote this passage?"

def make_record(text):
    return {"state": text, "questions": {"model": {
        "type": "choice", "instructions": INSTR, "criteria": {m: None for m in MODELS},
        "label": MODELS[0]}}}

from tqdm import tqdm
preds = []
for text in tqdm(test["text"].tolist()):
    probs = predictor(make_record(text))["probabilities"]["model"]
    preds.append(max(probs, key=probs.get))
print(pd.Series(preds).value_counts().to_string())"""),

    code("""\
sub = pd.DataFrame({"id": test["id"], "model": preds})
sub.to_parquet("/kaggle/working/submission.parquet", index=False)
print(sub.head())"""),
]

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}},
     "nbformat": 4, "nbformat_minor": 4}

if __name__ == "__main__":
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": "ibillxia/ntl-kev-sub",
        "title": "ntl kev sub",
        "code_file": "ntl_submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": ["ibillxia/ntl-qwen-base", "ibillxia/ntl-kev-run"],
        "competition_sources": ["name-that-llm"],
        "kernel_sources": [],
    }
    (HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", OUT, "and kernel-metadata.json")
