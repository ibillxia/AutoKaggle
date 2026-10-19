"""Render + push the inference-only XLM-R kernel (no training, so it fits the 2 h code limit).

    python build_xlmr_infer_notebook.py --model-subdir xlmr-ft [--push]

The fine-tuned checkpoint rides along in the same Kaggle dataset as the base weights, under
`xlmr-ft/`, so this kernel only has to locate the mounts and score test.csv.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATE = HERE / "kaggle_notebook" / "xlmr_infer_template.py"
OUT = ROOT / "kaggle" / "kernels" / "xlmr-watson-infer"

DATASETS = ["ibillxia/xlmr-nli", "ibillxia/xlmr-watson-ft"]
COMPETITION = "contradictory-my-dear-watson"


def render(model_subdir, maxlen, slug):
    code = TEMPLATE.read_text(encoding="utf-8")
    subs = {"__MAXLEN__": str(maxlen), "__MODEL_SUBDIR__": model_subdir}
    for k, v in subs.items():
        code = code.replace(k, v)
    left = [k for k in subs if k in code]
    if left:
        raise SystemExit("unsubstituted placeholders: %s" % left)
    compile(code, "main.py", "exec")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "main.py").write_text(code, encoding="utf-8")
    meta = {
        "id": slug,
        "title": slug.split("/")[-1].replace("-", " "),
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
    print("rendered", OUT, " model_subdir=%s maxlen=%s" % (model_subdir, maxlen))
    return OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-subdir", default="xlmr-watson-ft")
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--slug", default="ibillxia/xlmr-watson-infer")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    out = render(a.model_subdir, a.maxlen, a.slug)
    if a.push:
        cli = str(Path(sys.executable).parent / "kaggle.exe")
        r = subprocess.run([cli, "kernels", "push", "-p", str(out)], capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:])


if __name__ == "__main__":
    main()
