"""Render + push the Kaggle training kernel (train on GPU, then score test.csv in the same run).

    python build_watson_train_notebook.py --budget 95 --push
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATE = HERE / "kaggle_notebook" / "watson_train_template.py"
OUT = ROOT / "kaggle" / "kernels" / "watson-train"

DATASETS = ["ibillxia/xlmr-nli", "ibillxia/watson-train-data"]
COMPETITION = "contradictory-my-dear-watson"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", default="kev_train_huge.jsonl")
    ap.add_argument("--freeze-below", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=5e-5)
    ap.add_argument("--layer-decay", type=float, default=0.85)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--maxlen", type=int, default=192)
    ap.add_argument("--budget", type=int, default=95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-model", type=int, default=1)
    ap.add_argument("--slug", default="ibillxia/watson-train")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    code = TEMPLATE.read_text(encoding="utf-8")
    subs = {
        "__DATA_FILE__": a.data_file,
        "__FREEZE_BELOW__": str(a.freeze_below),
        "__EPOCHS__": str(a.epochs),
        "__LR__": repr(a.lr),
        "__HEAD_LR__": repr(a.head_lr),
        "__LAYER_DECAY__": repr(a.layer_decay),
        "__BATCH__": str(a.batch),
        "__ACCUM__": str(a.accum),
        "__MAXLEN__": str(a.maxlen),
        "__BUDGET__": str(a.budget),
        "__SEED__": str(a.seed),
        "__SAVE_MODEL__": str(a.save_model),
    }
    for k, v in subs.items():
        code = code.replace(k, v)
    left = [k for k in subs if k in code]
    if left:
        raise SystemExit("unsubstituted placeholders: %s" % left)
    compile(code, "main.py", "exec")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "main.py").write_text(code, encoding="utf-8")
    meta = {
        "id": a.slug,
        "title": a.slug.split("/")[-1].replace("-", " "),
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": DATASETS,
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("rendered", OUT)
    if a.push:
        cli = str(Path(sys.executable).parent / "kaggle.exe")
        r = subprocess.run([cli, "kernels", "push", "-p", str(OUT)], capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:])


if __name__ == "__main__":
    main()
