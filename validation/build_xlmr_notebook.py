"""Render + push the XLM-R Kaggle kernel for contradictory-my-dear-watson.

    python build_xlmr_notebook.py --epochs 3 --use-pool 1 [--push]

Renders validation/kaggle_notebook/xlmr_kernel_template.py into kaggle/kernels/xlmr-watson/ with a
kernel-metadata.json that mounts our xlmr-nli dataset (weights + on-distribution pool) and the
competition, internet off.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATE = HERE / "kaggle_notebook" / "xlmr_kernel_template.py"
OUT_ROOT = ROOT / "kaggle" / "kernels"

DATASET = "ibillxia/xlmr-nli"
COMPETITION = "contradictory-my-dear-watson"


def render(epochs, lr, batch, accum, maxlen, use_pool, seed, pool_name, model_dir, slug, en_repeat,
           grad_ckpt=1, freeze_below=-1):
    code = TEMPLATE.read_text(encoding="utf-8")
    subs = {
        "__EPOCHS__": str(epochs),
        "__LR__": str(lr),
        "__BATCH__": str(batch),
        "__ACCUM__": str(accum),
        "__MAXLEN__": str(maxlen),
        "__GRAD_CKPT__": "True" if grad_ckpt else "False",
        "__FREEZE_BELOW__": str(freeze_below),
        "__USE_POOL__": "True" if use_pool else "False",
        "__EN_REPEAT__": str(en_repeat),
        "__SEED__": str(seed),
        "__POOL__": pool_name,
        "__MODEL_DIR__": model_dir,
    }
    for k, v in subs.items():
        code = code.replace(k, v)
    left = [k for k in subs if k in code]
    if left:
        raise SystemExit("unsubstituted placeholders: %s" % left)
    compile(code, "main.py", "exec")          # fail here rather than on Kaggle
    out = OUT_ROOT / "xlmr-watson"
    out.mkdir(parents=True, exist_ok=True)
    (out / "main.py").write_text(code, encoding="utf-8")
    meta = {
        "id": slug,
        "title": slug.split("/")[-1].replace("-", " "),
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [DATASET],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("rendered", out, "  epochs=%s use_pool=%s pool=%s" % (epochs, bool(use_pool), pool_name or "-"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--freeze-below", type=int, default=-1)
    ap.add_argument("--use-pool", type=int, default=1)
    ap.add_argument("--en-repeat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-name", default="")
    ap.add_argument("--model-dir", default="", help="absolute path inside the Kaggle mount (optional)")
    ap.add_argument("--slug", default="ibillxia/xlmr-watson")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    out = render(a.epochs, a.lr, a.batch, a.accum, a.maxlen, a.use_pool, a.seed,
                 a.pool_name, a.model_dir, a.slug, a.en_repeat, a.grad_ckpt, a.freeze_below)
    if a.push:
        cli = str(Path(sys.executable).parent / "kaggle.exe")
        r = subprocess.run([cli, "kernels", "push", "-p", str(out)], capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:])


if __name__ == "__main__":
    main()
