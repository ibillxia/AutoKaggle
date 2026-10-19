"""Fine-tune a kev checkpoint on a labelled task's train split (the `--data` route).

kev ships its training stack (`kev.train`) so the same architecture that produced the released
checkpoints can be re-fit on a competition's own labels. This wrapper just assembles the argv and
hands it to `kev.train.main()`.

    python train_kev.py --task disaster --epochs 2

Notes on the two Windows shims this needs: `validation/_wincompat` supplies `fcntl` (used by
kev.suite for file locks) and `resource` (used by kev.device / kev.train for rlimit + peak RSS),
neither of which exists on Windows.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

sys.path.insert(0, str(HERE / "_wincompat"))          # fcntl + resource shims
sys.path.insert(0, str(ROOT / "third_party" / "kev"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

# the localised copies point `head.pt`'s base at models/<name> instead of a HF repo id,
# so training never tries to re-download the backbone
BASE_FOR = {
    "kev-0.8b": ROOT / "models" / "Qwen3.5-0.8B-Base",
    "kev-4b": ROOT / "models" / "Qwen3.5-4B-Base",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["imdb", "ag_news", "sms_spam", "disaster", "nli", "xnli"])
    ap.add_argument("--data", default="", help="explicit labelled jsonl; overrides --task's default split")
    ap.add_argument("--size", default="kev-0.8b", choices=sorted(BASE_FOR))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", default="")
    ap.add_argument("--init_from", default="", help="delta mode: warm-start LoRA + head from a released kev checkpoint")
    ap.add_argument("--head_lr", type=float, default=0.0)
    ap.add_argument("--extra", default="", help="extra kev.train flags, space separated")
    a = ap.parse_args()

    data = Path(a.data) if a.data else HERE / "data" / f"{a.task}_train.jsonl"
    if not data.exists():
        sys.exit(f"no train split at {data}; pass --data or run prep_data.py first")
    out = Path(a.out) if a.out else ROOT / "runs" / f"{a.size}-ft-{a.task}"

    argv = [
        "kev.train",
        "--base", str(BASE_FOR[a.size]),
        "--data", str(data),
        "--epochs", str(a.epochs),
        "--lora", str(a.lora),
        "--accum", str(a.accum),
        "--lr", str(a.lr),
        "--head_lr", str(a.head_lr),
        "--dtype", a.dtype,
        "--weights_dtype", a.dtype,
        "--device", a.device,
        "--out", str(out),
    ] + (["--init_from", a.init_from] if a.init_from else []) + a.extra.split()
    print(" ".join(argv), flush=True)

    from kev import train
    sys.argv = argv
    train.main()


if __name__ == "__main__":
    main()
