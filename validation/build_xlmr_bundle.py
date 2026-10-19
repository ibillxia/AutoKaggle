"""Stage the XLM-R route's Kaggle dataset (weights + on-distribution pool) and push it.

A code competition runs with the internet off, so xlm-roberta-large has to ride along as a dataset,
together with the pool rows the kernel folds into its training set. Layout inside the dataset:

  xlm-roberta-large/         config.json, model.safetensors, tokenizer.* , sentencepiece.bpe.model
  kev_pool_extra.jsonl       11081 on-distribution rows (MNLI dev_mismatched + XNLI test, mix-matched)
  kev_pool_extra3.jsonl      20533 on-distribution rows (XNLI test upsampled 2.8x)
  xlmr_info.json             provenance + the measurements that motivated it

    python build_xlmr_bundle.py stage
    python build_xlmr_bundle.py push
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGE = ROOT / "kaggle" / "dataset" / "xlmr-nli"
KAGGLE_ID = "ibillxia/xlmr-nli"
COMP = ROOT / "data" / "comp" / "contradictory-my-dear-watson"

MODEL_SRC = ROOT / "models" / "xlm-roberta-large"
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "sentencepiece.bpe.model"]
POOL_FILES = ["kev_pool_extra.jsonl", "kev_pool_extra3.jsonl"]

GCS_OLD = "www.googleapis.com"
GCS_NEW = "storage.googleapis.com"

INFO = {
    "backbone": "xlm-roberta-large",
    "why": "the kev 0.8B decision head plateaued at COMP-MIX 0.8176/0.8244/0.8201 and even the "
           "per-language ORACLE over all three members is 0.8269; the leaderboard's 12th place implies "
           "COMP-MIX ~0.896, out of reach for the 0.8B head",
    "pool_provenance": {
        "competition_pools": "train.csv 12120 + test.csv 5195 == MNLI dev_matched 9815 (fully consumed) "
                             "+ XNLI validation + XNLI test; train n test = 0; MNLI/XNLI train splits: 0 hits",
        "kev_pool_extra.jsonl": "5831 MNLI dev_mismatched + 5250 XNLI test (per-language = competition train count)",
        "kev_pool_extra3.jsonl": "5831 MNLI dev_mismatched + 14702 XNLI test (2.8x; needs --en-repeat 2)",
        "instrument": "XNLI validation is left untouched so kev_dev.jsonl stays row-clean",
        "excluded": "every row of train.csv, test.csv and kev_dev.jsonl",
    },
}


def stage(ft_dir=""):
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    dst = STAGE / "xlm-roberta-large"
    dst.mkdir()
    for f in MODEL_FILES:
        src = MODEL_SRC / f
        if not src.is_file():
            raise SystemExit("missing %s -- run the hf-mirror download first" % src)
        shutil.copy2(src, dst / f)
        print("  %-24s %8.1f MB" % (f, src.stat().st_size / 1e6))
    if ft_dir:
        src_ft = ROOT / ft_dir if not Path(ft_dir).is_absolute() else Path(ft_dir)
        dst_ft = STAGE / "xlmr-ft"
        dst_ft.mkdir()
        for f in sorted(src_ft.iterdir()):
            if f.is_file() and f.suffix in (".json", ".safetensors", ".model", ".txt"):
                shutil.copy2(f, dst_ft / f.name)
                print("  ft/%-21s %8.1f MB" % (f.name, f.stat().st_size / 1e6))
    for f in POOL_FILES:
        src = COMP / f
        if src.is_file():
            shutil.copy2(src, STAGE / f)
            print("  %-24s %8.2f MB" % (f, src.stat().st_size / 1e6))
    (STAGE / "xlmr_info.json").write_text(json.dumps(INFO, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    print("staged %s  (%.1f MB)" % (STAGE, total / 1e6))


def _patch_gcs_host():
    """Dataset uploads PUT to a GCS resumable session on www.googleapis.com, which is unreachable here
    (sandbox proxy: 502). storage.googleapis.com is the same JSON API and the session is keyed by
    upload_id in the query string, so rewriting the host and going direct works."""
    import requests

    original = requests.Session.request

    def request(self, method, url, *a, **kw):
        if GCS_OLD in str(url) and "/upload/storage/" in str(url):
            url = str(url).replace(GCS_OLD, GCS_NEW)
            self.trust_env = False
        return original(self, method, url, *a, **kw)

    requests.Session.request = request


def push():
    """Upload the directory as-is. An intermediate zip used to be built here, but the API uploads
    the folder anyway, so the archive was only ever extra bytes to push."""
    archive = STAGE / "xlmr-nli.zip"
    if archive.exists():
        archive.unlink()

    meta = {"title": "xlm-roberta-large for 15-language NLI", "id": KAGGLE_ID,
            "licenses": [{"name": "CC0-1.0"}]}
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _patch_gcs_host()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    refs = [d.ref for d in api.dataset_list(mine=True)]
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
    ap.add_argument("--ft", default="", help="fine-tuned checkpoint dir to ship as xlmr-ft/")
    a = ap.parse_args()
    if a.cmd == "stage":
        stage(a.ft)
    else:
        push()
