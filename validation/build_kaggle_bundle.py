"""Stage the files a Kaggle notebook needs to run kev offline, then (optionally) push them as a Dataset.

The notebook cannot download anything (code competitions run with the internet off), so everything kev needs
that is not already a public Kaggle Dataset/Model has to ride along:

  kev/                        the `kev` package source (importable from the dataset root)
  runs/<name>/head.pt         pointer head + temperatue, with `base` rewritten to the mounted base dataset
  runs/<name>/adapter_*.json  the LoRA adapter

Usage:
  python build_kaggle_bundle.py stage            # writes kaggle/dataset/kev-bundle
  python build_kaggle_bundle.py push             # kaggle datasets create/version
"""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGE = ROOT / "kaggle" / "dataset" / "kev-bundle"
KAGGLE_ID = "ibillxia/kev-bundle"

GCS_OLD = "www.googleapis.com"
GCS_NEW = "storage.googleapis.com"

# the public dataset that already holds the Qwen3.5-0.8B backbone on Kaggle
KAGGLE_BASE = "/kaggle/input/qwen35-official-08b-weights"

# name inside the bundle -> local run directory
SOURCES = {
    "kev-0.8b-base": ROOT / "models" / "kev-0.8b-local",
    "kev-0.8b-nli": ROOT / "runs" / "kev-0.8b-ft-watson-delta",
    "kev-0.8b-watson-full": ROOT / "runs" / "kev-0.8b-ft-watson-full",
    "kev-0.8b-llm": ROOT / "runs" / "kev-0.8b-ft-llm-delta",
    "kev-0.8b-llm-wide": ROOT / "runs" / "kev-0.8b-ft-llm-wide",
    "kev-0.8b-watson-ext": ROOT / "runs" / "kev-0.8b-ft-watson-ext",
}
DEFAULT_RUNS = list(SOURCES)


def stage(runs=None):
    runs = runs or DEFAULT_RUNS
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    # 1) the kev package
    shutil.copytree(ROOT / "third_party" / "kev" / "kev", STAGE / "kev",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 2) checkpoints, with `base` repointed at the Kaggle-mounted backbone. Every run ships in every version: a
    # notebook rerun pinned to "latest" must still find whatever `--run` its kernel was rendered with.
    kept = []
    for name in runs:
        src = SOURCES[name]
        if not (src / "head.pt").is_file():
            print(f"  {name}: SKIPPED, {src} has no head.pt yet")
            continue
        kept.append(name)
        dst = STAGE / "runs" / name
        dst.mkdir(parents=True)
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            if (src / f).is_file():
                shutil.copy2(src / f, dst / f)
            else:
                print(f"  {name}: no {f} (full-weight checkpoint?)")
        head = torch.load(src / "head.pt", map_location="cpu", weights_only=False)
        old = head.get("base")
        head["base"] = KAGGLE_BASE
        head["base_revision"] = None
        torch.save(head, dst / "head.pt")
        print(f"  {name}: base {old!r} -> {KAGGLE_BASE!r}  (T={head.get('temperature')})")

    # 3) a marker so the notebook can find things without guessing
    (STAGE / "bundle_info.json").write_text(json.dumps({
        "kev_base": KAGGLE_BASE,
        "runs": sorted(kept),
    }, indent=2), encoding="utf-8")

    total = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    print(f"staged {STAGE}  ({total/1e6:.1f} MB)")


def _patch_gcs_host():
    """Dataset uploads PUT to a GCS resumable session on www.googleapis.com, which is unreachable here
    (sandbox proxy: 502 Bad Gateway). storage.googleapis.com is the same GCS JSON API and the session is
    keyed by upload_id in the query string, so rewriting the host and going direct works — same fix as
    validation/kaggle_submit.py."""
    import requests

    original = requests.Session.request

    def request(self, method, url, *a, **kw):
        if GCS_OLD in str(url) and "/upload/storage/" in str(url):
            url = str(url).replace(GCS_OLD, GCS_NEW)
            self.trust_env = False
        return original(self, method, url, *a, **kw)

    requests.Session.request = request


def push():
    # one flat archive: `kaggle datasets create` skips directories, and the notebook unzips it anyway
    archive = STAGE / "kev-bundle.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(STAGE.rglob("*")):
            if f.is_file() and f not in (archive, STAGE / "dataset-metadata.json"):
                z.write(f, f.relative_to(STAGE))
    print(f"archive {archive} ({archive.stat().st_size/1e6:.1f} MB)")

    meta = {"title": "kev bundle code head adapter", "id": KAGGLE_ID,
            "licenses": [{"name": "CC0-1.0"}]}
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _patch_gcs_host()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    existing = api.dataset_list(mine=True)
    refs = [d.ref for d in existing]
    try:
        if KAGGLE_ID in refs:
            print(api.dataset_create_version(str(STAGE), "update", quiet=False, dir_mode="skip"))
        else:
            print(api.dataset_create_new(str(STAGE), public=False, quiet=False, dir_mode="skip"))
    except Exception as exc:  # noqa: BLE001
        print("push failed:", type(exc).__name__, str(exc)[:800])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "push"])
    ap.add_argument("--runs", default="", help="comma-separated bundle run names; default only the released head")
    a = ap.parse_args()
    runs = [r.strip() for r in a.runs.split(",") if r.strip()] or DEFAULT_RUNS
    if a.cmd == "stage":
        stage(runs)
    else:
        push()
