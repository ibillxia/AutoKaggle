"""Stage + push the fallback ("predictions as dataset") submission kernel.

    python build_probs_notebook.py stage   # upload submissions/mix_full_big.csv as a dataset
    python build_probs_notebook.py push    # push the tiny kernel that copies it to submission.csv
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATE = HERE / "kaggle_notebook" / "watson_probs_template.py"
OUT = ROOT / "kaggle" / "kernels" / "xlmr-watson-write"
STAGE = ROOT / "kaggle" / "dataset" / "watson-ens-probs"
PRED = ROOT / "submissions" / "mix_full_big.csv"
DATASET = "ibillxia/watson-ens-probs"
COMPETITION = "contradictory-my-dear-watson"
SLUG = "ibillxia/xlmr-watson-write"

CLI = str(Path(sys.executable).parent / "kaggle.exe")


def run(args):
    print(" ".join(args), flush=True)
    r = subprocess.run(args, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
    return r


def stage():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copy2(PRED, STAGE / "predictions.csv")
    (STAGE / "dataset-metadata.json").write_text(json.dumps({
        "title": "watson ensemble predictions",
        "id": DATASET,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2), encoding="utf-8")
    run([CLI, "datasets", "create", "-p", str(STAGE)])
    print("staged", STAGE)


def push():
    code = TEMPLATE.read_text(encoding="utf-8")
    compile(code, "main.py", "exec")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "main.py").write_text(code, encoding="utf-8")
    (OUT / "kernel-metadata.json").write_text(json.dumps({
        "id": SLUG,
        "title": "xlmr watson write",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [DATASET],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
    }, indent=2), encoding="utf-8")
    run([CLI, "kernels", "push", "-p", str(OUT)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "push"])
    a = ap.parse_args()
    stage() if a.cmd == "stage" else push()


if __name__ == "__main__":
    main()
