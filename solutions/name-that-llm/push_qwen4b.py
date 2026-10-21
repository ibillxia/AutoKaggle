# Push the staged ntl-qwen4b dataset (8.8 GB weights + training embeddings) with GCS host rewrite.
import json
import sys
from pathlib import Path

GCS_OLD = "www.googleapis.com"
GCS_NEW = "storage.googleapis.com"
ROOT = Path("E:/Bill/AI/AutoKaggle")
STAGE = ROOT / "kaggle" / "dataset" / "ntl-qwen4b"
DS_ID = "ibillxia/ntl-qwen4b"


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
    meta = {"title": "ntl-qwen4b", "id": DS_ID,
            "licenses": [{"name": "other"}],
            "description": "Qwen3.5-4B-Base weights + 7-layer mean-pooled train embeddings for the name-that-llm solution"}
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    patch_gcs_host()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    refs = [d.ref for d in api.dataset_list(mine=True)]
    if DS_ID in refs:
        print(api.dataset_create_version(str(STAGE), "weights + train embeddings", quiet=False, dir_mode="skip"))
    else:
        print(api.dataset_create_new(str(STAGE), public=False, quiet=False, dir_mode="skip"))


if __name__ == "__main__":
    main()
