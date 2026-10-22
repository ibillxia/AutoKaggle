"""Build ibillxia/ntl-hybrid: fork604's 0.604 wrapper + the five 14B folds that lifted ntl-ts to 0.601.

The 0.604 notebook is a wrapper: its first 370 lines are the two-stage LoRA stack whose `LORAS`
dict holds exactly the two members ntl-ts also started from (lora8b_v1 0.226, lora14b_v1 0.429),
then it mixes in the four-Gaussian stack at 0.85/0.15 and applies two data-validated guards
(em-dash invariant, dense-prompt Hungarian).

ntl-ts reached 0.601 from the same two-stage base by adding five Qwen3-14B two-stage folds
(w=0.07 each, twin 0.35 x that) and two prvsiyan seeds.  The folds drop into this wrapper's
`LORAS` dict unchanged -- same `find("<name>/meta.json")` layout, same `lora_features` loader,
same 5120/40 base -- while `prvsiyan` needs a different loader and is left out of v1.

So the hybrid = 0.601's fold averaging on the two-stage side + 0.604's Gaussian mix and guards.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_NB = os.path.join(HERE, "fork604b", "ntl-fork604b.ipynb")
OUT_DIR = os.path.join(HERE, "hybrid")
DST_NB = os.path.join(OUT_DIR, "ntl-hybrid.ipynb")

FOLD_W, TWIN_RATIO = 0.07, 0.35   # ntl-ts: 5 folds x 0.07, twin weight fixed at 0.35 of the plain one
FOLDS = ["ts14b_f%d" % i for i in range(5)]

OLD = '''    "lora14b_v1": dict(w=0.429, twin_w=0.150, bs=8,  base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),
}'''

NEW_HEAD = '''    "lora14b_v1": dict(w=0.429, twin_w=0.150, bs=8,  base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),
    # --- ntl-hybrid: the five two-stage 14B folds of zhuyu2/name-that-llm-ts-lora (the 0.576 -> 0.601 gain).
    # Same trunk (qwen3/5120/40, keep=27 from each meta.json) so the trunk is reloaded per fold but the
    # weights come from the page cache; each fold only contributes a small correction on top of lora14b_v1.
'''

EXTRA_DATASETS = ["zhuyu2/name-that-llm-ts-lora"]


def main():
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    assert len(nb["cells"]) == 1, "fork604 is a single-cell notebook; adjust the injection"
    src = nb["cells"][0]["source"]
    assert src.count(OLD) == 1, "LORAS dict changed upstream"

    entries = "".join(
        '    "%s": dict(w=%.3f, twin_w=%.4f, bs=8, base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),\n'
        % (name, FOLD_W, FOLD_W * TWIN_RATIO) for name in FOLDS)
    src = src.replace(OLD, NEW_HEAD + entries + "}", 1)
    compile(src, "<hybrid>", "exec")          # fail here, not after two hours of GPU
    nb["cells"][0]["source"] = src
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(nb, open(DST_NB, "w", encoding="utf-8"), indent=1)

    meta = json.load(open(os.path.join(HERE, "fork604b", "kernel-metadata.json"), encoding="utf-8"))
    meta["id"] = "ibillxia/ntl-hybrid"
    meta["title"] = "ntl hybrid gauss plus folds"
    meta["code_file"] = "ntl-hybrid.ipynb"
    meta["is_private"] = True
    for d in EXTRA_DATASETS:
        if d not in meta["dataset_sources"]:
            meta["dataset_sources"].append(d)
    json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    print("wrote", DST_NB)
    print("datasets", len(meta["dataset_sources"]), "kernels", len(meta["kernel_sources"]),
          "models", len(meta["model_sources"]))
    print("folds added:", FOLDS, "w=%.2f twin_w=%.4f" % (FOLD_W, FOLD_W * TWIN_RATIO))


if __name__ == "__main__":
    main()
