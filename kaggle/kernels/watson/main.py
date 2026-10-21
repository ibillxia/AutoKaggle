"""Kaggle kernel body for a code-competition submission driven by a local kev checkpoint.

`validation/build_notebook.py` substitutes the __PLACEHOLDER__s and writes this next to a
kernel-metadata.json, then pushes it. It runs on Kaggle with the internet off, so everything it
needs is mounted:

  /kaggle/input/datasets/<owner>/<slug>/       ours (kev package + checkpoint) and the public backbone
  /kaggle/input/competitions/<slug>/test.csv   the competition's own test split

Kaggle's actual mount layout (probed, 2026-10) is:
  /kaggle/input/competitions/<slug>/...        competition sources
  /kaggle/input/datasets/<owner>/<slug>/...    dataset sources

Two question types are wired up:
  * choice  (3 criteria: NLI relation, or which of two LLM answers won)
  * noul    (yes/no)
"""
import csv
import json
import math
import os
import shutil
import sys
import time

WORK = "/kaggle/working"
COMP = "contradictory-my-dear-watson"
QUESTION = "relation"
KIND = "choice"                 # "choice" | "noul"
INSTRUCTIONS = "What is the logical relationship between the premise and the hypothesis?"
CRITERIA = json.loads("""[\"entailment\", \"neutral\", \"contradiction\"]""")   # [] for noul
LABEL_MAP = json.loads("""[0, 1, 2]""")  # criterion -> competition column value
RUN_NAME = "kev-0.8b-watson-ext"
# Ensemble members, in order: [{"run": name, "prompt_chars": p, "resp_chars": r, "weight": w}, ...].
# One member (the common case) is just a plain single-model submission; several members are averaged in
# probability space, which is the only lever that improved nlp-getting-started monotonically on the real LB
# (0.82623 -> 0.82745 -> 0.83113 for 1 -> 2 -> 3 members). Widths travel per member because a head only ever
# saw the prefix it was trained on.
RUN_SPEC = json.loads("""[{\"run\": \"kev-0.8b-watson-ext\", \"prompt_chars\": 300, \"resp_chars\": 600, \"weight\": 1.0}]""")
OUT_MODE = "label"          # "label" = one argmax column | "probs" = one column per criterion
PRED_COL = "prediction"          # column name for OUT_MODE == "label"
SUBMISSION = os.path.join(WORK, "submission.csv")

t_start = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - t_start, msg), flush=True)


# ---------------------------------------------------------------- locate + unpack the bundle
# Kaggle mounts a dataset at /kaggle/input/<slug>/ and auto-extracts uploaded archives (into a folder
# named after the archive), so the bundle root is wherever `kev/checkpoint.py` actually landed.
import glob  # noqa: E402


def find_bundle_root():
    hits = sorted(glob.glob("/kaggle/input/**/kev/checkpoint.py", recursive=True))
    if hits:
        return os.path.dirname(os.path.dirname(hits[0]))
    raise SystemExit("kev bundle not found under /kaggle/input: %s" % sorted(os.listdir("/kaggle/input")))


def find_competition_file(name):
    """Competition files live at /kaggle/input/competitions/<slug>/<name> (probed)."""
    for pat in ("/kaggle/input/competitions/*/%s" % name,
                "/kaggle/input/*/%s" % name,
                "/kaggle/input/**/%s" % name):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    raise SystemExit("competition file %r not found; /kaggle/input -> %s"
                     % (name, sorted(os.listdir("/kaggle/input"))))


KB = find_bundle_root()
log("bundle root %s -> %s" % (KB, sorted(os.listdir(KB))))
sys.path.insert(0, KB)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

# Kaggle's image ships torchao 0.10.0, which peft 0.20 refuses to use; its probe raises ImportError
# instead of returning False, and peft calls that probe on every LoRA dispatch, so adapter loading dies.
# kev never asks for torchao quantisation, so neutralise the probe.
try:
    import peft.import_utils as _piu
    _piu._torchao_available = False
    _piu.is_torchao_available = lambda version=None: False
    import peft.tuners.lora.torchao as _pltao
    _pltao.is_torchao_available = lambda version=None: False
    log("neutralised peft's torchao probe (image has an incompatible torchao)")
except Exception as exc:  # noqa: BLE001
    log("torchao probe shim skipped: %s" % exc)

from kev.checkpoint import LoadOptions  # noqa: E402
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_PACKED, SERVE_MAX_STATE  # noqa: E402
from kev.predictors import LocalPredictor  # noqa: E402

# kev.suite.SERVING_CONTEXT, spelled out so the kernel does not pull in the whole suite module
SERVING_CONTEXT = {"max_state": SERVE_MAX_STATE, "max_branch": SERVE_MAX_BRANCH,
                   "max_packed": SERVE_MAX_PACKED, "truncate": False}

log("torch %s cuda=%s n=%d" % (torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()))

RUN_SRC = os.path.join(KB, "runs", RUN_SPEC[0]["run"])
log("run source %s -> %s" % (RUN_SRC, sorted(os.listdir(RUN_SRC))))
log("members: %s" % [(m["run"], m["prompt_chars"], m["resp_chars"], m["weight"]) for m in RUN_SPEC])


def find_base():
    for cfg in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        try:
            if json.load(open(cfg, encoding="utf-8")).get("model_type") == "qwen3_5":
                return os.path.dirname(cfg)
        except Exception:  # noqa: BLE001
            pass
    raise SystemExit("Qwen3.5 backbone not found under /kaggle/input")


# /kaggle/input is read-only and head.pt records where the backbone lived when it was trained, so work on
# a copy: patch `base` to whatever the backbone is mounted as here. One copy per ensemble member.
BASE_DIR = find_base()


def stage_run(name):
    src = os.path.join(KB, "runs", name)
    dst = os.path.join(WORK, "run_" + name)
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith((".json", ".safetensors", ".pt")):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    head = torch.load(os.path.join(dst, "head.pt"), map_location="cpu", weights_only=False)
    if head.get("base") != BASE_DIR:
        head["base"] = BASE_DIR
        head["base_revision"] = None
        torch.save(head, os.path.join(dst, "head.pt"))
        log("  %s: head.pt base -> %s" % (name, BASE_DIR))
    return dst


RUN = os.path.join(WORK, "run")          # kept for the single-member case / older logs
MEMBER_DIRS = [stage_run(m["run"]) for m in RUN_SPEC]
RUN = MEMBER_DIRS[0]


# ---------------------------------------------------------------- read the test split
TEST_CSV = find_competition_file("test.csv")
log("test.csv -> %s" % TEST_CSV)


def read_rows():
    with open(TEST_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_state(r, pch, rch):
    p = (r.get('premise') or '').strip()
    h = (r.get('hypothesis') or '').strip()
    return 'Premise: %s\n\nHypothesis: %s' % (p, h)


# One record set per member: the state prefix differs with the member's truncation width.
records_by_member, ids = [], []
_raw = read_rows()
for m in RUN_SPEC:
    recs = []
    for r in _raw:
        # `label` is only a placeholder here (probabilities do not depend on it) but kev's serving path
        # requires it to be a real option key, so hand it the first criterion.
        q = {"type": KIND, "instructions": INSTRUCTIONS, "src": COMP,
             "label": CRITERIA[0] if KIND == "choice" else False}
        if CRITERIA:
            q["criteria"] = {k: k for k in CRITERIA}
        recs.append({"state": build_state(r, m["prompt_chars"], m["resp_chars"]),
                     "questions": {QUESTION: q}})
    records_by_member.append(recs)
ids = [r["id"] for r in _raw]
records = records_by_member[0]
log("%d test rows x %d member(s)" % (len(records), len(RUN_SPEC)))

device = "cuda" if torch.cuda.is_available() else "cpu"
# Turing (T4, sm_75) has no bf16 tensor cores, so bf16 there is emulated and slower than fp32.
# fp32 is also the path kev's reported numbers use, so it is the safe default everywhere else.
if device == "cuda":
    cap = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if cap[0] >= 8 else torch.float32
    log("device %s sm_%d%d -> dtype %s" % (torch.cuda.get_device_name(0), cap[0], cap[1], dtype))
    opts = LoadOptions(dtype=dtype, backend="torch")
else:
    opts = LoadOptions()
t = time.time()
PREDICTORS = [LocalPredictor(d, device, opts, context=SERVING_CONTEXT) for d in MEMBER_DIRS]
log("loaded %d member(s) in %.1fs (T=%s)" % (
    len(PREDICTORS), time.time() - t, [round(p.checkpoint.meta.temperature, 4) for p in PREDICTORS]))

# ---------------------------------------------------------------- score
COLS = ["id", PRED_COL] if OUT_MODE == "label" else ["id"] + (CRITERIA or ["target"])

# Optional vector scaling: kev's serving path applies the checkpoint temperature only, so a per-class bias
# fitted offline (validation/fit_temperature.py --bias) has to be applied here. p_c <- p_c * exp(b_c), renorm.
# Empty dict = identity, which is what every run trained before this switch used.
CLASS_BIAS = {}


def apply_bias(probs):
    if not CLASS_BIAS:
        return probs
    w = {k: v * math.exp(CLASS_BIAS.get(k, 0.0)) for k, v in probs.items()}
    z = sum(w.values()) or 1.0
    return {k: v / z for k, v in w.items()}


_WEIGHTS = [float(m.get("weight", 1.0)) for m in RUN_SPEC]
_WSUM = sum(_WEIGHTS) or 1.0


def predict_row(i):
    """Weighted probability average over the members. Diversity is real here: separate fine-tunes disagree
    on 8-13% of rows (measured on nlp-getting-started), and averaging only ever helped there."""
    if len(PREDICTORS) == 1:
        p = PREDICTORS[0](records_by_member[0][i])["probabilities"][QUESTION]
    else:
        acc, keys = {}, None
        for w, pred, recs in zip(_WEIGHTS, PREDICTORS, records_by_member):
            p = pred(recs[i])["probabilities"][QUESTION]
            keys = keys or list(p)
            for k, v in p.items():
                acc[k] = acc.get(k, 0.0) + w * v
        p = {k: acc[k] / _WSUM for k in keys}
    return apply_bias(p)


def dump():
    """Write the submission as it stands. Called periodically so a timeout still leaves a valid file."""
    with open(SUBMISSION, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


rows, probs_rows, fallbacks, t0 = [], [], 0, time.time()
for i, rec in enumerate(records):
    try:
        probs = predict_row(i)
        key = max(probs, key=probs.get)
        value = LABEL_MAP[CRITERIA.index(key)] if CRITERIA else (1 if key == "true" else 0)
    except Exception as exc:  # never drop a submission row
        fallbacks += 1
        probs = {c: 1.0 / max(len(CRITERIA), 2) for c in CRITERIA} if CRITERIA else {"true": 0.0, "false": 1.0}
        key = CRITERIA[0] if CRITERIA else "false"
        value = LABEL_MAP[0] if CRITERIA else 0
        log("  ! row %d fell back: %s: %s" % (i, type(exc).__name__, exc))
    row = {"id": ids[i]}
    if OUT_MODE == "label":
        row[PRED_COL] = value
    elif CRITERIA:
        for c in CRITERIA:
            row[c] = round(float(probs.get(c, 0.0)), 6)
    else:
        row["target"] = value
    rows.append(row)
    probs_rows.append({"id": ids[i], "key": key, "p": probs})
    if (i + 1) % 200 == 0:
        log("  %d/%d  %.3f s/row" % (i + 1, len(records), (time.time() - t0) / (i + 1)))
        dump()

wall = time.time() - t0
dump()
with open(os.path.join(WORK, "probs.jsonl"), "w", encoding="utf-8") as f:
    for r in probs_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter  # noqa: E402
log("scored %d rows in %.0fs (%.3f s/row), fallbacks=%d" % (len(rows), wall, wall / len(rows), fallbacks))
log("argmax distribution: %s" % dict(sorted(Counter(r["key"] for r in probs_rows).items())))
if device == "cuda":
    log("peak VRAM %.2f GB" % (torch.cuda.max_memory_allocated() / 1024 ** 3))
log("wrote %s" % SUBMISSION)
print(open(SUBMISSION, encoding="utf-8").read()[:400])
