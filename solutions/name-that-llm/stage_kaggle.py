# Stage Kaggle datasets for the Name That LLM kev submission.
#   kaggle/dataset/ntl-qwen-base/  <- Qwen3.5-0.8B-Base weights (~1.7GB)
#   kaggle/dataset/ntl-kev-run/    <- fine-tuned checkpoint + kev source + inference assets
import json
import shutil
from pathlib import Path

ROOT = Path("E:/Bill/AI/AutoKaggle")
STAGE_BASE = ROOT / "kaggle" / "dataset" / "ntl-qwen-base"
STAGE_RUN = ROOT / "kaggle" / "dataset" / "ntl-kev-run"
RUN = ROOT / "runs" / "kev-0.8b-ft-ntl-v2"
KEV_SRC = ROOT / "third_party" / "kev" / "kev"

BASE_FILES = ["config.json", "model.safetensors-00001-of-00001.safetensors",
              "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt"]
RUN_FILES = ["head.pt", "adapter_config.json", "adapter_model.safetensors",
             "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]

def stage():
    STAGE_BASE.mkdir(parents=True, exist_ok=True)
    for f in BASE_FILES:
        src = ROOT / "models" / "Qwen3.5-0.8B-Base" / f
        dst = STAGE_BASE / f
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
    (STAGE_BASE / "dataset-metadata.json").write_text(json.dumps({
        "title": "ntl qwen base", "id": "ibillxia/ntl-qwen-base",
        "licenses": [{"name": "CC0-1.0"}]}, indent=2), encoding="utf-8")

    STAGE_RUN.mkdir(parents=True, exist_ok=True)
    for f in RUN_FILES:
        shutil.copy2(RUN / f, STAGE_RUN / f)
    # vendor kev source (pure python) so the offline notebook can import it
    ks = STAGE_RUN / "kev_src"
    if ks.exists():
        shutil.rmtree(ks)
    shutil.copytree(KEV_SRC, ks, ignore=shutil.ignore_patterns("__pycache__"))
    (STAGE_RUN / "dataset-metadata.json").write_text(json.dumps({
        "title": "ntl kev run", "id": "ibillxia/ntl-kev-run",
        "licenses": [{"name": "CC0-1.0"}]}, indent=2), encoding="utf-8")
    print("staged:", STAGE_BASE, [p.name for p in STAGE_BASE.iterdir()])
    print("staged:", STAGE_RUN, [p.name for p in STAGE_RUN.iterdir()])

if __name__ == "__main__":
    stage()
