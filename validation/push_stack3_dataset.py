"""Upload the already-staged submissions/stack3_lang.csv as the dataset ibillxia/watson-stack3.

The stock `kaggle datasets create` uploads to www.googleapis.com, which is unreachable here, so
the same host rewrite used by kaggle_submit.py / push_model_dataset.py is applied first.
"""
import json
from pathlib import Path

GCS_OLD = "www.googleapis.com"
GCS_NEW = "storage.googleapis.com"
ROOT = Path("E:/Bill/AI/AutoKaggle")
STAGE = ROOT / "kaggle" / "dataset" / "watson-stack3"
DS_ID = "ibillxia/watson-stack3"


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
    (STAGE / "dataset-metadata.json").write_text(json.dumps({
        "title": "watson stack3 predictions", "id": DS_ID,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2), encoding="utf-8")
    print("staged:", [p.name for p in STAGE.iterdir()])

    patch_gcs_host()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    refs = [d.ref for d in api.dataset_list(mine=True)]
    if DS_ID in refs:
        print(api.dataset_create_version(str(STAGE), "update stack3", quiet=False, dir_mode="skip"))
    else:
        print(api.dataset_create_new(str(STAGE), public=False, quiet=False, dir_mode="skip"))


if __name__ == "__main__":
    main()
