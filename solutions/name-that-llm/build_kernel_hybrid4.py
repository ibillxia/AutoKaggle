# Build hybrid4 = fork604b (0.604 base) + poolv2_8b (w=0.070) appended to LORAS.
# Single-variable test of the pool_v2-trained member on the strong base.
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_NB = os.path.join(HERE, "fork604b", "ntl-fork604b.ipynb")
OUT_DIR = os.path.join(HERE, "hybrid4")
NAME = "poolv2_8b"
ENTRY = ('    # --- hybrid4: pool_v2-trained 8B member (stage-1 pool 34,940 -> 70,034 rows,\n'
         '    # stage2a holdout 0.4852 vs old lora8b_v1 0.4370), appended as a small correction.\n'
         '    "poolv2_8b": dict(w=0.070, twin_w=0.0245, bs=8, '
         'base=dict(model_type="qwen3", hidden_size=4096, num_hidden_layers=36)),\n')
OLD = '''    "lora14b_v1": dict(w=0.429, twin_w=0.150, bs=8,  base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),
}'''
NEW = '''    "lora14b_v1": dict(w=0.429, twin_w=0.150, bs=8,  base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),
''' + ENTRY + '}'

def main():
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    assert len(nb["cells"]) == 1
    src = nb["cells"][0]["source"]
    assert src.count(OLD) == 1, "LORAS dict changed"
    src = src.replace(OLD, NEW, 1)
    compile(src, "<hybrid4>", "exec")   # fail here, not after hours of GPU
    nb["cells"][0]["source"] = src
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(nb, open(os.path.join(OUT_DIR, "ntl-hybrid4.ipynb"), "w", encoding="utf-8"), indent=1)

    meta = json.load(open(os.path.join(HERE, "fork604b", "kernel-metadata.json"), encoding="utf-8"))
    meta["id"] = "ibillxia/ntl-hybrid4-poolv2"      # MUST match title slug (409 lesson)
    meta["title"] = "ntl hybrid4 poolv2"
    meta["code_file"] = "ntl-hybrid4.ipynb"
    if "ibillxia/ntl-pool-v2-lora" not in meta["dataset_sources"]:
        meta["dataset_sources"].append("ibillxia/ntl-pool-v2-lora")
    json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    print("hybrid4 written:", OUT_DIR)

if __name__ == "__main__":
    main()
