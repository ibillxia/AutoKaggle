"""Render + push the inference-only XLM-R kernel (no training, so it fits the 2 h code limit).

    python build_xlmr_infer_notebook.py --spec xlmr-watson-ft-big:1.382:0.5,xlmr-watson-ft-full:1.231:0.5 [--push]

`--spec` is a comma-separated list of `dataset-name[:temperature[:weight]]`. Each member must live
in a Kaggle dataset that is mounted under /kaggle/input; the kernel tempers each member's softmax
with its dev-fitted temperature, then averages.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATE = HERE / "kaggle_notebook" / "xlmr_infer_ens_template.py"
OUT = ROOT / "kaggle" / "kernels" / "xlmr-watson-infer"

BASE_DATASET = "ibillxia/xlmr-nli"
COMPETITION = "contradictory-my-dear-watson"


def parse_spec(spec):
    members = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        bits = [b.strip() for b in part.split(":")]
        name = bits[0]
        temp = float(bits[1]) if len(bits) > 1 and bits[1] else 1.0
        weight = float(bits[2]) if len(bits) > 2 and bits[2] else 1.0
        members.append((name, temp, weight))
    if not members:
        raise SystemExit("empty --spec")
    return members


def render(members, maxlen, slug, gpu=True):
    code = TEMPLATE.read_text(encoding="utf-8")
    spec_lit = "[" + ", ".join("(%r, %r, %r)" % (n, t, w) for n, t, w in members) + "]"
    subs = {"__MAXLEN__": str(maxlen), "__MODEL_SPEC__": spec_lit}
    for k, v in subs.items():
        code = code.replace(k, v)
    left = [k for k in subs if k in code]
    if left:
        raise SystemExit("unsubstituted placeholders: %s" % left)
    compile(code, "main.py", "exec")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "main.py").write_text(code, encoding="utf-8")

    sources = [BASE_DATASET]
    for name, _, _ in members:
        ds = "ibillxia/" + name
        if ds not in sources:
            sources.append(ds)
    meta = {
        "id": slug,
        "title": slug.split("/")[-1].replace("-", " "),
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": bool(gpu),
        "enable_internet": False,
        "dataset_sources": sources,
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("rendered", OUT, " members=%s maxlen=%s sources=%s" % (members, maxlen, sources))
    return OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="xlmr-watson-ft-big:1.0")
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--slug", default="ibillxia/xlmr-watson-infer")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force a CPU session (skips the 2-GPU-session cap)")
    a = ap.parse_args()
    members = parse_spec(a.spec)
    out = render(members, a.maxlen, a.slug, gpu=not a.cpu)
    if a.push:
        cli = str(Path(sys.executable).parent / "kaggle.exe")
        r = subprocess.run([cli, "kernels", "push", "-p", str(out)], capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:])


if __name__ == "__main__":
    main()
