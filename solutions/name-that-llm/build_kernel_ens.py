"""Build ntl-ens.ipynb: add prvsiyan's seed variants as extra LoRA members.

Why: prvsiyan/ntl-two-stage-lora-adapters-v1 is an independent re-training of the same
recipe (Qwen3-8B-Base L24, rank 16, 8-layer pooled, plain + centred heads) on a *different*
public pool (their in-style pool, 38,642 passages). Honest prompt-grouped CV of their
full_s0: 0.5675 -- the same as the author's 0.569. So these members are the same quality as
the current ensemble but decorrelated by training data, which is exactly what an ensemble
needs.

They ship their own peft-free implementation (adapters/lora_cls.py) with
load_for_inference / predict_probs, and their meta.json carries its own label order
(alphabetical) plus bos=1 / max_len=384 conventions that must not be mixed with the
nishantdahal path. We therefore call their code for their checkpoints and reorder columns.

Base-model pinning matters: fork592's find_base(spec) returns the first sorted config.json
matching model_type/hidden_size, and /kaggle/input/**/8b-base/... sorts before .../8b/...
so merely attaching 8b-base would silently swap the base under the existing 8B member.
Each member therefore gets an explicit dir.

Variants written to /kaggle/working (submission.parquet = PICK):
  submission_full.parquet     svc + 2 lora + 2 extra, twin, then Sinkhorn
  submission_notwin.parquet   same, no centred twin
  submission_nosink.parquet   same, no Sinkhorn
  submission_plain.parquet    same, neither
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "ens")
os.makedirs(OUT_DIR, exist_ok=True)

SRC_NB = os.path.join(HERE, "fork592", "ntl-fork592.ipynb")
DST_NB = os.path.join(OUT_DIR, "ntl-ens.ipynb")

# extra members: adapter subdir -> blend weight. Two seeds of the same recipe/data are
# highly correlated, so the pair gets roughly the weight of one independent 8B member.
EXTRA_W = {"q8b_L24_ctr/full_s0": 0.11, "q8b_L24_ctr/full_s1": 0.11}

BASE_PIN = r'''
# ---- deterministic base resolution (see the generator docstring) ----
BASE_8B_DIR = find("transformers/8b/1/config.json")
BASE_8B_DIR = os.path.dirname(BASE_8B_DIR)
BASE_14B_DIR = os.path.dirname(find("transformers/14b/1/config.json"))
BASE_8B_BASE_DIR = os.path.dirname(find("transformers/8b-base/1/config.json"))
print("bases:", BASE_8B_DIR, BASE_14B_DIR, BASE_8B_BASE_DIR, flush=True)
LORAS["lora8b_v1"]["dir"] = BASE_8B_DIR
LORAS["lora14b_v1"]["dir"] = BASE_14B_DIR
'''

EXTRA_CELL = r'''
# ==================== extra members: prvsiyan q8b_L24_ctr ====================
# Their own peft-free implementation; their conventions (bos=1, max_len=384,
# alphabetically ordered labels) are handled entirely inside their code.
#
# This cell runs BEFORE the 24/27-layer nishantdahal members: their loader builds a
# 10.6GB fp16 trunk in one go and cannot share a card, so it needs a clean device.
# Whatever is left afterwards is released before the next member loads.
import importlib
import sys

ADAPT_ROOT = os.path.dirname(find("adapters/lora_cls.py"))
sys.path.insert(0, ADAPT_ROOT)
lora_cls = importlib.import_module("lora_cls")

EXTRA = {
%s
}


def free_gpu():
    """Index of the device that can hold a 10.6GB fp16 trunk right now (garbage included:
    a dispatched model leaves reference cycles behind, so collect before measuring)."""
    gc.collect()
    torch.cuda.empty_cache()
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    i = max(range(len(free)), key=free.__getitem__)
    print(f"extra member on cuda:{i} free={[round(f / 2 ** 30, 1) for f in free]}GiB", flush=True)
    return i


for rel, w in EXTRA.items():
    t_m = time.time()
    # their build_model picks dtype with `torch.float16 if device == "cuda" else float32`,
    # so a device string like "cuda:1" silently means fp32 and OOMs a 14.5GB T4. Select the
    # card with set_device and pass exactly "cuda".
    torch.cuda.set_device(free_gpu())
    adir = os.path.join(ADAPT_ROOT, rel)
    m2, tok2, meta2 = lora_cls.load_for_inference(adir, base_dir=BASE_8B_BASE_DIR, device="cuda")
    P, bad = lora_cls.predict_probs(m2, tok2, test.text.tolist(), meta2, device="cuda", bs=16)
    order = [meta2["labels"].index(l) for l in LABELS]
    p = np.clip(P[:, order], 1e-9, 1)
    p = p / p.sum(1, keepdims=True)
    LOGIT += w * np.log(p)
    del m2, tok2, P, p
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{rel} done w={w} bad={int(bad)} ({time.time()-t_m:.0f}s, total {time.time()-T0:.0f}s)", flush=True)
del m2, tok2, meta2
gc.collect()
torch.cuda.empty_cache()
'''

FINAL_CELL = r'''
def sinkhorn(P):
    P = P.copy()
    if len(P) < 50:          # the 3-row public test.csv: balancing is meaningless
        return P
    target = np.full(K, len(P) / K)
    for _ in range(100):
        P *= target / P.sum(0)
        P /= P.sum(1, keepdims=True)
    return P


VARIANTS = {
    "submission_full.parquet":   sinkhorn(softmax(LOGIT_BOTH)),
    "submission_notwin.parquet": sinkhorn(softmax(LOGIT)),
    "submission_nosink.parquet": softmax(LOGIT_BOTH),
    "submission_plain.parquet":  softmax(LOGIT),
}
for fname, P in VARIANTS.items():
    pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[P.argmax(1)]}).to_parquet(fname, index=False)
    print(fname, "->", np.array(LABELS)[P.argmax(1)].tolist()[:3], flush=True)

# Kaggle only accepts a file named exactly submission.parquet.
PICK = "submission_full.parquet"
P = VARIANTS[PICK]
pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[P.argmax(1)]}).to_parquet("submission.parquet", index=False)
print(f"picked {PICK} -> submission.parquet")
print(f"done in {time.time() - T0:.0f}s")
'''


def main():
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    code_cells = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    assert len(code_cells) == 5, code_cells

    lora_idx = None
    for i in code_cells:
        src = "".join(nb["cells"][i]["source"])
        if "for ckpt_name, mc in LORAS.items():" in src:
            lora_idx = i
            old = '    Z, heads = lora_features(ckpt, find_base(mc["base"]), test.text.tolist(), mc["bs"])'
            assert old in src, "loop line changed upstream"
            src = src.replace(
                old,
                '    Z, heads = lora_features(ckpt, mc.get("dir") or find_base(mc["base"]), test.text.tolist(), mc["bs"])',
            )
            # the extra members need a clean 10.6GB block on cuda:0 after the 14B member
            # has spanned both cards, so release the cache between members
            stale = "    del Z\n    gc.collect()"
            assert stale in src
            src = src.replace(stale, stale + "\n    torch.cuda.empty_cache()", 1)
            # keep the centred twin's contribution separate so the final cell can ablate it:
            # fork592 adds it straight into LOGIT, which leaves 'with twin' and 'without twin'
            # indistinguishable (and references an undefined LOGIT_BOTH downstream)
            assert "LOGIT[twin_rows] +=" in src
            src = src.replace("LOGIT[twin_rows] +=", "TWIN_LOGIT[twin_rows] +=")
            anchor = "for ckpt_name, mc in LORAS.items():"
            src = src.replace(anchor, "TWIN_LOGIT = np.zeros_like(LOGIT)\n\n" + anchor, 1)
            nb["cells"][i]["source"] = src.splitlines(keepends=True)
        elif "def lora_features(" in src:
            # a device_map="auto" trunk (the 14B) leaves reference cycles behind, so a bare
            # `del model` does not return its 18GB until the cyclic collector runs -- the
            # previous run OOM'd in the extra-members cell for exactly this reason
            stale = "    del model\n    torch.cuda.empty_cache()"
            assert stale in src, "cleanup block changed upstream"
            src = src.replace(stale, "    del model\n    gc.collect()\n    torch.cuda.empty_cache()", 1)
            nb["cells"][i]["source"] = src.splitlines(keepends=True)
        elif "submission.parquet" in src:
            nb["cells"][i]["source"] = FINAL_CELL.strip("\n").splitlines(keepends=True)
        elif "LORAS = {" in src:
            # cell 1: LABELS / K / LORAS / find are all defined here; append the pin at the
            # very end, since find() is defined below K.
            assert "def find(pattern):" in src, "find() moved; adjust the injection point"
            nb["cells"][i]["source"] = (src + "\n" + BASE_PIN).splitlines(keepends=True)

    assert lora_idx is not None
    extra = "\n".join(f'    "{k}": {v},' for k, v in EXTRA_W.items())
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": (EXTRA_CELL % extra).strip("\n").splitlines(keepends=True),
    }
    # before the nishantdahal loop: the extra members need a whole free card
    nb["cells"].insert(lora_idx, cell)
    json.dump(nb, open(DST_NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written", DST_NB)

    meta = json.load(open(os.path.join(HERE, "fork592", "kernel-metadata.json"), encoding="utf-8"))
    meta.update({
        "id": "ibillxia/ntl-ens",
        "title": "ntl ens",
        "code_file": "ntl-ens.ipynb",
        "dataset_sources": ["nishantdahal/ntl-two-stage-lora", "prvsiyan/ntl-two-stage-lora-adapters-v1"],
        "model_sources": [
            "qwen-lm/qwen-3/transformers/8b/1",
            "qwen-lm/qwen-3/transformers/14b/1",
            "qwen-lm/qwen-3/transformers/8b-base/1",
        ],
    })
    json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    print("written metadata")


if __name__ == "__main__":
    main()
