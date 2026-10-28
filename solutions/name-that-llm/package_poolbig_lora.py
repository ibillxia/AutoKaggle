# Publish the poolbig stage-2b checkpoint as ibillxia/ntl-poolbig-lora.
#
# Layout lesson (cost us hybrid3 ERROR + one hybrid4 v1 run):
#   Kaggle unzips an uploaded zip at the dataset root. With dir_mode='zip' the folder structure is
#   flattened, so `find("poolbig_8b/adapter/adapter_config.json")` in the kernel fails.
#   Correct recipe: build a zip whose internal paths already carry the `poolbig_8b/` prefix, upload
#   that zip with dir_mode='skip'. Verified working on ibillxia/ntl-pool-v2-lora (v2).
import json, os, shutil, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "poolbig-out", "ntl8b_poolbig_v1")
STAGE = os.path.join(HERE, "nt-run-logs", "poolbig_lora_zip_stage")
INNER = "poolbig_8b"          # must equal the LORAS key in the kernel
DS_ID = "ibillxia/ntl-poolbig-lora"
DS_TITLE = "NTL poolbig lora"


def need(rel):
    p = os.path.join(SRC, rel)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        sys.exit("missing/empty: %s" % p)
    return p


def main():
    need("adapter/adapter_config.json")
    need("adapter/adapter_model.safetensors")
    need("heads.safetensors")
    need("meta.json")

    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)

    zpath = os.path.join(STAGE, INNER + ".zip")
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for root, _, files in os.walk(SRC):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, SRC).replace("\\", "/")
                z.write(full, INNER + "/" + rel)
                n += 1
    print("zip:", zpath, z.namelist()[:6], "files:", n, os.path.getsize(zpath) // 1024 // 1024, "MB")

    meta = {
        "title": DS_TITLE,
        "id": DS_ID,
        "licenses": [{"name": "CC-BY-4.0"}],
        "description": "Two-stage LoRA member (qwen3 8B base, keep=24) trained on pool_big = "
                       "pool_v2 (70,034 rows) U crtproduction open pool, md5-deduped -> 259,006 rows / "
                       "86,381 prompts. stage2a holdout balanced acc 0.5389 (pool_v2 baseline 0.4852). "
                       "Folder poolbig_8b/ follows the LORAS find() layout.",
    }
    with open(os.path.join(STAGE, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("stage ready:", STAGE)


if __name__ == "__main__":
    main()
