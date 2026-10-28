# Build hybrid5 = fork604b (0.604 base) + poolbig_8b (w=0.070), REPLACING the poolv2_8b member.
# Derived from hybrid4 (which already carries the weight-aware find_base / find_checkpoint patch that
# v1 of hybrid4 proved necessary after Kaggle changed the dataset mount layout).
#
# Why replace instead of append: poolbig's stage-1 pool is a strict superset of pool_v2's
# (pool_v2 70,034 U open-pool -> 259,006 rows), so keeping both would add two members of the same
# lineage -- exactly the "same-pool retrain kills blend diversity" failure reported in discussion 753177.
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "hybrid4")
SRC_NB = os.path.join(SRC_DIR, "ntl-hybrid4.ipynb")
OUT_DIR = os.path.join(HERE, "hybrid5")

OLD = ('    # --- hybrid4: pool_v2-trained 8B member (stage-1 pool 34,940 -> 70,034 rows,\n'
       '    # stage2a holdout 0.4852 vs old lora8b_v1 0.4370), appended as a small correction.\n'
       '    "poolv2_8b": dict(w=0.070, twin_w=0.0245, bs=8, '
       'base=dict(model_type="qwen3", hidden_size=4096, num_hidden_layers=36)),\n')
NEW = ('    # --- hybrid5: big-pool 8B member (stage-1 pool 34,940 -> 259,006 rows; pool_v2 U open-pool\n'
       '    # after md5 dedup), stage2a holdout 0.5389 vs pool_v2 0.4852 / old lora8b_v1 0.4370.\n'
       '    "poolbig_8b": dict(w=0.070, twin_w=0.0245, bs=8, '
       'base=dict(model_type="qwen3", hidden_size=4096, num_hidden_layers=36)),\n')


def main():
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    assert len(nb["cells"]) == 1, len(nb["cells"])
    src = nb["cells"][0]["source"]
    assert src.count(OLD) == 1, "hybrid4 LORAS entry changed"
    src = src.replace(OLD, NEW, 1)
    assert "poolv2_8b" not in src, "stale poolv2_8b reference left behind"
    # the weight-aware patch must ride along, otherwise this crashes on the new mount layout
    assert "_has_weights" in src and "_has_w" in src, "weight guard patch missing in source notebook"
    compile(src, "<hybrid5>", "exec")
    nb["cells"][0]["source"] = src
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(nb, open(os.path.join(OUT_DIR, "ntl-hybrid5.ipynb"), "w", encoding="utf-8"), indent=1)

    meta = json.load(open(os.path.join(SRC_DIR, "kernel-metadata.json"), encoding="utf-8"))
    meta["id"] = "ibillxia/ntl-hybrid5-poolbig"      # MUST equal the title slug (409 lesson)
    meta["title"] = "ntl hybrid5 poolbig"
    meta["code_file"] = "ntl-hybrid5.ipynb"
    ds = [d for d in meta["dataset_sources"] if d != "ibillxia/ntl-pool-v2-lora"]
    if "ibillxia/ntl-poolbig-lora" not in ds:
        ds.append("ibillxia/ntl-poolbig-lora")
    meta["dataset_sources"] = ds
    json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    print("hybrid5 written:", OUT_DIR)
    print("dataset_sources:", meta["dataset_sources"])


if __name__ == "__main__":
    main()
