"""Build ntl-ablate.ipynb from the fork592 notebook, emitting several blend variants as files.

Hypothesis under test: the author's LB 0.592 predates the centred twin + Sinkhorn additions
published in the current notebook; those two tricks may be neutral or harmful on the real test
layout. One kernel run writes four candidate submissions so we can spend submission quota only
on the variants we want to compare.

Variants written to /kaggle/working:
  submission_full.parquet     svc + lora(plain) + twin, then Sinkhorn   == current notebook
  submission_notwin.parquet   svc + lora(plain), then Sinkhorn
  submission_nosink.parquet   svc + lora(plain) + twin, no Sinkhorn
  submission_plain.parquet    svc + lora(plain) only
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "ablate")
os.makedirs(OUT_DIR, exist_ok=True)

SRC_NB = os.path.join(HERE, "fork592", "ntl-fork592.ipynb")
DST_NB = os.path.join(OUT_DIR, "ntl-ablate.ipynb")

LORA_CELL = r'''
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(test.text.tolist())
D = np.clip(1 - (Xt @ Xt.T).toarray(), 0, None)
np.fill_diagonal(D, 0)
groups = fcluster(linkage(squareform(D, checks=False), method="average"), TWIN["thresh"], criterion="distance") \
    if len(test) >= 2 else np.ones(len(test), dtype=int)
size = np.bincount(groups)[groups]
twin_rows = (size >= TWIN["min_size"]) & (size <= TWIN["max_size"])
del D
print(f"topic clusters: {int((size == 1).sum())} singletons, {int(twin_rows.sum())} passages in clusters of 2-18", flush=True)

LOGIT_TWIN = np.zeros_like(LOGIT)   # kept separate so we can ablate the centred twin
for ckpt_name, mc in LORAS.items():
    ckpt = os.path.dirname(find(f"{ckpt_name}/meta.json"))
    Z, heads = lora_features(ckpt, find_base(mc["base"]), test.text.tolist(), mc["bs"])
    LOGIT += mc["w"] * np.log(np.clip(softmax(Z @ heads["plain.weight"].T + heads["plain.bias"]), 1e-9, 1))
    if twin_rows.any():
        Zc = Z.copy()
        for g in np.unique(groups[twin_rows]):
            Zc[groups == g] -= Zc[groups == g].mean(0)
        centred = Zc[twin_rows] @ heads["centered.weight"].T + heads["centered.bias"]
        LOGIT_TWIN[twin_rows] += mc["twin_w"] * np.log(np.clip(softmax(centred), 1e-9, 1))
    del Z
    gc.collect()
    print(f"{ckpt_name} done ({time.time() - T0:.0f}s)", flush=True)

LOGIT_BOTH = LOGIT + LOGIT_TWIN
print(f"twin applied to {int(twin_rows.sum())}/{len(test)} rows", flush=True)
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
    "submission_full.parquet":   sinkhorn(softmax(LOGIT_BOTH)),   # current notebook
    "submission_notwin.parquet": sinkhorn(softmax(LOGIT)),        # no centred twin
    "submission_nosink.parquet": softmax(LOGIT_BOTH),             # no Sinkhorn
    "submission_plain.parquet":  softmax(LOGIT),                  # neither
}
for fname, P in VARIANTS.items():
    pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[P.argmax(1)]}).to_parquet(fname, index=False)
    print(fname, "->", np.array(LABELS)[P.argmax(1)].tolist()[:3], flush=True)

# Kaggle rejects any submission file not named exactly submission.parquet, so the chosen
# variant is mirrored under that name; the others stay around for inspection.
PICK = "submission_plain.parquet"
P = VARIANTS[PICK]
pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[P.argmax(1)]}).to_parquet("submission.parquet", index=False)
print(f"picked {PICK} -> submission.parquet", flush=True)
print(f"done in {time.time() - T0:.0f}s")
'''


def main():
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    code_cells = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    assert len(code_cells) == 5, code_cells
    for i in code_cells:
        src = "".join(nb["cells"][i]["source"])
        if "for ckpt_name, mc in LORAS.items():" in src:
            nb["cells"][i]["source"] = LORA_CELL.strip("\n").splitlines(keepends=True)
        elif "submission.parquet" in src:
            nb["cells"][i]["source"] = FINAL_CELL.strip("\n").splitlines(keepends=True)
    json.dump(nb, open(DST_NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written", DST_NB)

    meta = json.load(open(os.path.join(HERE, "fork592", "kernel-metadata.json"), encoding="utf-8"))
    meta.update({
        "id": "ibillxia/ntl-ablate",
        "title": "ntl ablate",
        "code_file": "ntl-ablate.ipynb",
    })
    json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    print("written metadata", meta)


if __name__ == "__main__":
    main()
