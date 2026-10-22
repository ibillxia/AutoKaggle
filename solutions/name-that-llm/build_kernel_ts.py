"""Build ntl-ts.ipynb: the tried ensemble plus five two-stage 14B folds.

Why these folds: zhuyu2/name-that-llm-ts-lora publishes five Qwen3-14B two-stage LoRA
checkpoints (fold 0-4 of the host train split, same public pool, same recipe as the
nishantdahal members). Their layout is byte-for-byte compatible with fork592's loader --
same `adapter/adapter_model.safetensors`, same `heads.safetensors` with norms.0-7 plus
plain/centered, same label order -- so they drop straight into lora_features() with the
14B base and no new code. Five folds share 80% of their training rows, so together they
behave like roughly one strong independent member (~+1 effective member), but the
disagreement is real and free.

Members and weights. The published 8B/14B/SVC weights came from the author's OOF fit; the
new folds have no OOF here, so they get a fixed budget (0.07 each, twins at the recipe's
0.35 ratio, which both published members happen to share: 0.079/0.226 = 0.150/0.429 =
0.35). The prvsiyan pair keeps its 0.11 per seed.

Every member's log-probabilities are kept separately, so the final cell can emit several
blends from one expensive pass:
  submission_all.parquet   every member (PICK)
  submission_noprv.parquet the same without the prvsiyan seeds
  submission_old.parquet   svc + 8B + 14B, i.e. a reproduction of the 0.576 submission
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "ts")
os.makedirs(OUT_DIR, exist_ok=True)

SRC_NB = os.path.join(HERE, "fork592", "ntl-fork592.ipynb")
DST_NB = os.path.join(OUT_DIR, "ntl-ts.ipynb")

TS_FOLDS = [f"ts14b_f{i}" for i in range(5)]
TS_W = 0.07
TS_TWIN_RATIO = 0.35
PRV = {"q8b_L24_ctr/full_s0": 0.11, "q8b_L24_ctr/full_s1": 0.11}

BASE_PIN = r'''
# ---- deterministic base resolution (see the generator docstring) ----
BASE_8B_DIR = os.path.dirname(find("transformers/8b/1/config.json"))
BASE_14B_DIR = os.path.dirname(find("transformers/14b/1/config.json"))
BASE_8B_BASE_DIR = os.path.dirname(find("transformers/8b-base/1/config.json"))
print("bases:", BASE_8B_DIR, BASE_14B_DIR, BASE_8B_BASE_DIR, flush=True)
LORAS["lora8b_v1"]["dir"] = BASE_8B_DIR
LORAS["lora14b_v1"]["dir"] = BASE_14B_DIR

MEM = {}        # member name -> its weighted-free log-probabilities (n, K)
TWIN_MEM = {}   # same, for the centred twin head (only twin_rows are meaningful)
'''

PRV_CELL = r'''
# ==================== prvsiyan seeds: independent data, same recipe ====================
# Their own peft-free implementation; bos=1, max_len=384 and alphabetical labels are
# handled inside their code. Runs before the big members because their loader builds a
# whole 10.6GB fp16 trunk at once and needs a card to itself.
import importlib
import sys

ADAPT_ROOT = os.path.dirname(find("adapters/lora_cls.py"))
sys.path.insert(0, ADAPT_ROOT)
lora_cls = importlib.import_module("lora_cls")

PRV = {
%s
}


def free_gpu():
    """Index of the device that can hold a 10.6GB fp16 trunk right now (garbage included:
    a dispatched model leaves reference cycles behind, so collect before measuring)."""
    gc.collect()
    torch.cuda.empty_cache()
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    i = max(range(len(free)), key=free.__getitem__)
    print(f"prvsiyan member on cuda:{i} free={[round(f / 2 ** 30, 1) for f in free]}GiB", flush=True)
    return i


for rel, w in PRV.items():
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
    MEM["prv_" + rel.split("/")[-1]] = np.log(p)
    LOGIT += w * MEM["prv_" + rel.split("/")[-1]]
    del m2, tok2, P, p
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{rel} done w={w} bad={int(bad)} ({time.time()-t_m:.0f}s, total {time.time()-T0:.0f}s)", flush=True)
gc.collect()
torch.cuda.empty_cache()
'''

TS_CELL = r'''
# ==================== five two-stage 14B folds (zhuyu2/name-that-llm-ts-lora) ==========
# Drop-in for lora_features: identical adapter/heads layout, identical label order,
# keep=27 on Qwen3-14B-Base, same 8 pooled layers. max_len comes from their meta.json.
TS_ROOT = os.path.dirname(find("ts14b_f0/meta.json"))


def twin_logprobs(Z, groups, twin_rows, heads):
    """Centred head read out on the same-prompt differences -- the twin of the plain head."""
    Zc = Z.copy()
    for g in np.unique(groups[twin_rows]):
        Zc[groups == g] -= Zc[groups == g].mean(0)
    centred = Zc[twin_rows] @ heads["centered.weight"].T + heads["centered.bias"]
    return np.log(np.clip(softmax(centred), 1e-9, 1))


for name in TS_FOLDS:
    t_m = time.time()
    ckpt = os.path.join(TS_ROOT, name)
    Z, heads = lora_features(ckpt, BASE_14B_DIR, test.text.tolist(), 8)
    MEM[name] = np.log(np.clip(softmax(Z @ heads["plain.weight"].T + heads["plain.bias"]), 1e-9, 1))
    LOGIT += TS_W * MEM[name]
    if twin_rows.any():
        TWIN_MEM[name] = twin_logprobs(Z, groups, twin_rows, heads)   # twin_rows only
        TWIN_LOGIT[twin_rows] += (TS_TWIN_RATIO * TS_W) * TWIN_MEM[name]
    del Z, heads
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{name} done ({time.time()-t_m:.0f}s, total {time.time()-T0:.0f}s)", flush=True)
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


def blend(members, twins):
    L = sum(w * MEM[m] for m, w in members.items())
    if twins and twin_rows.any():
        T = np.zeros_like(L)
        for m, w in twins.items():
            T[twin_rows] += w * TWIN_MEM[m]
        L = L + T
    return sinkhorn(softmax(L))


W_BASE = {"svc": W_SVC, "lora8b_v1": 0.226, "lora14b_v1": 0.429}
TWIN_BASE = {"lora8b_v1": 0.079, "lora14b_v1": 0.150}
FOLDS = {n: TS_W for n in TS_FOLDS if n in MEM}
TWIN_FOLDS = {n: TS_TWIN_RATIO * TS_W for n in FOLDS}
PRVS = {n: 0.11 for n in MEM if n.startswith("prv_")}

VARIANTS = {
    "submission_all.parquet":   blend({**W_BASE, **FOLDS, **PRVS}, {**TWIN_BASE, **TWIN_FOLDS}),
    "submission_noprv.parquet": blend({**W_BASE, **FOLDS}, {**TWIN_BASE, **TWIN_FOLDS}),
    "submission_old.parquet":   blend(W_BASE, TWIN_BASE),
}
for fname, P in VARIANTS.items():
    pred = np.array(LABELS)[P.argmax(1)]
    pd.DataFrame({"id": test["id"].astype(str), "model": pred}).to_parquet(fname, index=False)
    print(fname, "->", pred.tolist()[:3], f"| {len(np.unique(pred))} distinct labels", flush=True)

PICK = "submission_all.parquet"
pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[VARIANTS[PICK].argmax(1)]}) \
    .to_parquet("submission.parquet", index=False)
print(f"picked {PICK} -> submission.parquet")
print(f"members: {sorted(MEM)}")
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
            # keep each member's log-probs so the final cell can emit several blends, and
            # keep the twin's share separate so it can be ablated or reweighted
            old_lp = '    LOGIT += mc["w"] * np.log(np.clip(softmax(Z @ heads["plain.weight"].T + heads["plain.bias"]), 1e-9, 1))'
            assert old_lp in src, "plain accumulation changed upstream"
            src = src.replace(
                old_lp,
                '    MEM[ckpt_name] = np.log(np.clip(softmax(Z @ heads["plain.weight"].T + heads["plain.bias"]), 1e-9, 1))\n'
                '    LOGIT += mc["w"] * MEM[ckpt_name]',
            )
            old_tw = '        LOGIT[twin_rows] += mc["twin_w"] * np.log(np.clip(softmax(centred), 1e-9, 1))'
            assert old_tw in src, "twin accumulation changed upstream"
            src = src.replace(
                old_tw,
                '        TWIN_MEM[ckpt_name] = np.log(np.clip(softmax(centred), 1e-9, 1))\n'
                '        TWIN_LOGIT[twin_rows] += mc["twin_w"] * TWIN_MEM[ckpt_name]',
            )
            stale = "    del Z\n    gc.collect()"
            assert stale in src
            src = src.replace(stale, stale + "\n    torch.cuda.empty_cache()", 1)
            anchor = "for ckpt_name, mc in LORAS.items():"
            src = src.replace(anchor, "TWIN_LOGIT = np.zeros_like(LOGIT)\n\n" + anchor, 1)
            src = src + "\n\n" + TS_CELL.strip("\n") + "\n"
            nb["cells"][i]["source"] = src.splitlines(keepends=True)
        elif "def lora_features(" in src:
            # a device_map="auto" trunk leaves reference cycles behind; a bare `del model`
            # does not hand the 18GB back until the cyclic collector runs
            stale = "    del model\n    torch.cuda.empty_cache()"
            assert stale in src, "cleanup block changed upstream"
            src = src.replace(stale, "    del model\n    gc.collect()\n    torch.cuda.empty_cache()", 1)
            nb["cells"][i]["source"] = src.splitlines(keepends=True)
        elif "P_svc = softmax(" in src:
            old = "LOGIT += W_SVC * np.log(np.clip(P_svc, 1e-9, 1))"
            assert old in src, "svc accumulation changed upstream"
            src = src.replace(
                old,
                'MEM["svc"] = np.log(np.clip(P_svc, 1e-9, 1))\nLOGIT += W_SVC * MEM["svc"]',
            )
            nb["cells"][i]["source"] = src.splitlines(keepends=True)
        elif "submission.parquet" in src:
            nb["cells"][i]["source"] = FINAL_CELL.strip("\n").splitlines(keepends=True)
        elif "LORAS = {" in src:
            assert "def find(pattern):" in src, "find() moved; adjust the injection point"
            nb["cells"][i]["source"] = (src + "\n" + BASE_PIN).splitlines(keepends=True)

    assert lora_idx is not None
    prv = "\n".join(f'    "{k}": {v},' for k, v in PRV.items())
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": (PRV_CELL % prv).strip("\n").splitlines(keepends=True),
    }
    nb["cells"].insert(lora_idx, cell)

    # constants the appended fold loop reads at run time
    tsconst = f"TS_FOLDS = {TS_FOLDS!r}\nTS_W = {TS_W}\nTS_TWIN_RATIO = {TS_TWIN_RATIO}\n\n"
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and "TWIN_LOGIT = np.zeros_like(LOGIT)" in "".join(c["source"]):
            src = "".join(c["source"]).replace("TWIN_LOGIT = np.zeros_like(LOGIT)",
                                               tsconst + "TWIN_LOGIT = np.zeros_like(LOGIT)", 1)
            nb["cells"][i]["source"] = src.splitlines(keepends=True)
            break

    json.dump(nb, open(DST_NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written", DST_NB)

    meta = json.load(open(os.path.join(HERE, "fork592", "kernel-metadata.json"), encoding="utf-8"))
    meta.update({
        "id": "ibillxia/ntl-ts",
        "title": "ntl ts",
        "code_file": "ntl-ts.ipynb",
        "dataset_sources": [
            "nishantdahal/ntl-two-stage-lora",
            "zhuyu2/name-that-llm-ts-lora",
            "prvsiyan/ntl-two-stage-lora-adapters-v1",
        ],
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
