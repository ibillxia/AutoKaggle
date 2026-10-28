# Upload nt-run-logs/poolbig_lora_zip_stage as dataset ibillxia/ntl-poolbig-lora.
# Same GCS host rewrite as push_pool_dataset.py: the upload create_url points at www.googleapis.com,
# which is unreachable here; storage.googleapis.com serves the identical JSON API and the upload_id
# lives in the query string, so swapping the host is safe.
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from push_pool_dataset import patch_gcs_host

STAGE = os.path.join(HERE, "nt-run-logs", "poolbig_lora_zip_stage")
DS_ID = "ibillxia/ntl-poolbig-lora"


def main():
    assert os.path.exists(os.path.join(STAGE, "poolbig_8b.zip")), "zip missing"
    assert os.path.exists(os.path.join(STAGE, "dataset-metadata.json")), "metadata missing"
    patch_gcs_host()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    refs = [d.ref for d in api.dataset_list(mine=True)]
    if DS_ID in refs:
        print(api.dataset_create_version(STAGE, "poolbig lora member", quiet=False, dir_mode="skip"))
    else:
        print(api.dataset_create_new(STAGE, public=False, quiet=False, dir_mode="skip"))


if __name__ == "__main__":
    main()
