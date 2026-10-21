# Push the staged ntl-qwen-base / ntl-kev-run datasets to Kaggle (host-rewritten GCS upload).
import json
from pathlib import Path

GCS_OLD = "www.googleapis.com"
GCS_NEW = "storage.googleapis.com"
ROOT = Path("E:/Bill/AI/AutoKaggle")
STAGES = [
    (ROOT / "kaggle" / "dataset" / "ntl-qwen-base", "ibillxia/ntl-qwen-base", "qwen3.5-0.8b-base weights for ntl"),
    (ROOT / "kaggle" / "dataset" / "ntl-kev-run", "ibillxia/ntl-kev-run", "kev-0.8b ft checkpoint + source for ntl"),
]


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
    patch_gcs_host()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    refs = [d.ref for d in api.dataset_list(mine=True)]
    for stage, ds_id, msg in STAGES:
        if ds_id in refs:
            print(ds_id, "-> create_version")
            print(api.dataset_create_version(str(stage), msg, quiet=False, dir_mode="skip"))
        else:
            print(ds_id, "-> create_new")
            print(api.dataset_create_new(str(stage), public=False, quiet=False, dir_mode="skip"))


if __name__ == "__main__":
    main()
