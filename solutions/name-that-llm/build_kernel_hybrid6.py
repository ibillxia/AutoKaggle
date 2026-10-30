# Build hybrid6 = fork604b (0.604) with its 8B member REPLACED by the poolbig-trained one.
#
# Rationale: hybrid5 (appending poolbig_8b at w=0.070) scored 0.599 -- appending a new member to a
# strong blend does not transfer. Discussion consensus is that a same-lineage UPGRADE (swap the old
# member trained on a small pool for the same recipe trained on a much bigger pool) is the variant
# that does transfer. lora8b_v1 was trained on 34,940 rows (holdout 0.4370); poolbig_8b is the same
# recipe on 259,006 rows (holdout 0.5389). Everything else stays bit-identical: w=0.226, twin_w=0.079,
# bs=16, the Gaussian stack, the 14B member, the label order.
#
# The swap uses an explicit PATH_OVERRIDE instead of renaming the LORAS key, so that
#   1) the `if ckpt_name in ('lora8b_v1', 'lora14b_v1')` trunk/Gaussian branch still matches, and
#   2) `find("poolbig_8b/meta.json")` has exactly one candidate (a renamed lora8b_v1 folder would
#      collide with the original dataset folder and make sorted(glob)[0] layout-dependent -- the
#      exact class of bug that killed hybrid4 v1).
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_NB = os.path.join(HERE, "hybrid4", "ntl-hybrid4.ipynb")   # already carries the weight-aware patches
OUT_DIR = os.path.join(HERE, "hybrid6")

POOLV2_ENTRY = '''
    # --- hybrid4: pool_v2-trained 8B member (stage-1 pool 34,940 -> 70,034 rows,
    # stage2a holdout 0.4852 vs old lora8b_v1 0.4370), appended as a small correction.
    "poolv2_8b": dict(w=0.070, twin_w=0.0245, bs=8, base=dict(model_type="qwen3", hidden_size=4096, num_hidden_layers=36)),'''

OLD_8B = '''    "lora8b_v1":  dict(w=0.226, twin_w=0.079, bs=16, base=dict(model_type="qwen3", hidden_size=4096, num_hidden_layers=36)),'''
NEW_8B = '''    # hybrid6: same recipe, stage-1 pool 34,940 -> 259,006 rows (holdout 0.4370 -> 0.5389).
    # The folder is poolbig_8b (see PATH_OVERRIDE); the key stays lora8b_v1 so the trunk/Gaussian
    # branch and its 0.10 twin weight apply unchanged.
    "lora8b_v1":  dict(w=0.226, twin_w=0.079, bs=16, base=dict(model_type="qwen3", hidden_size=4096, num_hidden_layers=36)),'''

OLD_RESOLVE = 'ckpt = os.path.dirname(find(f"{ckpt_name}/meta.json"))'
NEW_RESOLVE = 'ckpt = os.path.dirname(find(f"{PATH_OVERRIDE.get(ckpt_name, ckpt_name)}/meta.json"))'

OLD_TWIN = "LORAS = {  # checkpoint folder: blend weight, centred-twin weight, batch size, base model to look for"
NEW_TWIN = ('# hybrid6 member-1 folder override: the 8B slot is filled by the poolbig checkpoint.\n'
            'PATH_OVERRIDE = {"lora8b_v1": "poolbig_8b"}\n\n'
            'LORAS = {  # checkpoint folder: blend weight, centred-twin weight, batch size, base model to look for')


def main():
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    assert len(nb["cells"]) == 1
    src = nb["cells"][0]["source"]

    assert src.count(POOLV2_ENTRY) == 1, "poolv2_8b entry not found verbatim"
    src = src.replace(POOLV2_ENTRY, "", 1)

    assert src.count(OLD_8B) == 1, "lora8b_v1 entry not found verbatim"
    src = src.replace(OLD_8B, NEW_8B, 1)

    assert src.count(OLD_RESOLVE) == 1, "ckpt resolution line not found verbatim"
    src = src.replace(OLD_RESOLVE, NEW_RESOLVE, 1)

    assert src.count(OLD_TWIN) == 1, "LORAS header not found verbatim"
    src = src.replace(OLD_TWIN, NEW_TWIN, 1)

    compile(src, "<hybrid6>", "exec")   # fail here, not after hours of GPU

    nb["cells"][0]["source"] = src
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(nb, open(os.path.join(OUT_DIR, "ntl-hybrid6.ipynb"), "w", encoding="utf-8"), indent=1)

    meta = json.load(open(os.path.join(HERE, "fork604b", "kernel-metadata.json"), encoding="utf-8"))
    meta["id"] = "ibillxia/ntl-hybrid6-poolbigswap"     # MUST match title slug
    meta["title"] = "ntl hybrid6 poolbigswap"
    meta["code_file"] = "ntl-hybrid6.ipynb"
    dsets = [d for d in meta["dataset_sources"] if d != "ibillxia/ntl-pool-v2-lora"]
    if "ibillxia/ntl-poolbig-lora" not in dsets:
        dsets.append("ibillxia/ntl-poolbig-lora")
    meta["dataset_sources"] = dsets
    json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    print("hybrid6 written:", OUT_DIR)
    print("datasets:", meta["dataset_sources"])


if __name__ == "__main__":
    main()
