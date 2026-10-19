"""Push a fine-tuned checkpoint as its own private Kaggle dataset.

`dataset_create_version(dir_mode="skip")` silently skips new *folders*, so nesting the checkpoint
inside the existing xlmr-nli dataset does not work. A separate dataset keeps the upload to just the
one 2.2 GB file we actually changed.

    python push_model_dataset.py --dir runs/xlmr-fb12-big --id ibillxia/xlmr-watson-ft
"""
import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GCS_OLD = "www.googleapis.com"
GCS_NEW = "storage.googleapis.com"
KEEP = {".json", ".safetensors", ".model", ".txt", ".spm"}


def patch_gcs_host():
    import requests
    original = requests.Session.request

    def request(self, method, url, *a, **kw):
        if GCS_OLD in str(url) and "/upload/storage/" in str(url):
            url = str(url).replace(GCS_OLD, GCS_NEW)
            self.trust_env = False
        return original(self, method, url, *a, **kw)

    requests.Session.request = request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--stage", default="")
    a = ap.parse_args()

    src = Path(a.dir)
    if not src.is_absolute():
        src = ROOT / src
    stage = Path(a.stage) if a.stage else ROOT / "kaggle" / "dataset" / a.id.split("/")[-1]
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    total = 0
    for f in sorted(src.iterdir()):
        if f.is_file() and f.suffix in KEEP:
            shutil.copy2(f, stage / f.name)
            total += f.stat().st_size
            print("  %-26s %8.1f MB" % (f.name, f.stat().st_size / 1e6))
    print("staged %s (%.1f MB)" % (stage, total / 1e6))

    meta = {"title": "watson XLM-R large fine-tuned", "id": a.id,
            "licenses": [{"name": "CC0-1.0"}]}
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    patch_gcs_host()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    refs = [d.ref for d in api.dataset_list(mine=True)]
    try:
        if a.id in refs:
            print(api.dataset_create_version(str(stage), "update", quiet=False, dir_mode="skip"))
        else:
            print(api.dataset_create_new(str(stage), public=False, quiet=False, dir_mode="skip"))
    except Exception as exc:  # noqa: BLE001
        print("push failed:", type(exc).__name__, str(exc)[:800])


if __name__ == "__main__":
    main()
