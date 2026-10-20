"""Push the watson training corpus as a private Kaggle dataset (train JSONL + dev instrument).

    python push_train_dataset.py --id ibillxia/watson-train-data
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COMP = ROOT / "data/comp/contradictory-my-dear-watson"

FILES = ["kev_train_huge.jsonl", "kev_train_big.jsonl", "kev_dev.jsonl", "kev_dev.meta.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="ibillxia/watson-train-data")
    ap.add_argument("--force-new", action="store_true")
    a = ap.parse_args()

    stage = ROOT / "kaggle" / "dataset" / a.id.split("/")[-1]
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    total = 0
    for name in FILES:
        src = COMP / name
        if not src.is_file():
            print("skip missing", name)
            continue
        shutil.copy2(src, stage / name)
        total += src.stat().st_size
        print("  %-24s %8.1f MB" % (name, src.stat().st_size / 1e6))
    print("staged %s (%.1f MB)" % (stage, total / 1e6))

    meta = {"title": "watson NLI training corpus",
            "id": a.id, "licenses": [{"name": "CC0-1.0"}]}
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    sys.path.insert(0, str(HERE))
    from push_model_dataset import patch_gcs_host  # noqa: E402
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
