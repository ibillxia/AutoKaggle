"""LMSYS human-preference inference on vanilla transformers.

The public 0.8304 notebook depends on `human_pref`, which imports xformers and custom
triton kernels.  The current Kaggle image (python 3.13, torch 2.11, transformers 5.16)
ships neither, so we rebuild the identical architecture straight from the checkpoint
weights:

    base transformer (HF Gemma2Model / LlamaModel, sdpa)
      -> hidden state of the last non-pad token
      -> Dropout -> Linear(h, h/2) -> Dropout -> GELU -> Linear(h/2, 3)

Both RMSNorm conventions line up with HF (Llama: init ones used directly; Gemma2: init
zeros used as 1+w), so the state dict ports over with only the `model.` prefix stripped
and the fused `gate_up_proj` split in two.

MODE="test" : score the hidden test set under a hard 9h wall-clock cap and write
              submission.csv.  The cap is the whole design constraint: 25k rows x 2
              members x 2 response orders is ~100k forward passes, which does not fit,
              so the second order is only run when the measured first pass leaves room.

Three hard-won details, each of which cost a multi-hour run:
  * read shards with plain read() -- mmap on the FUSE mount runs at 1.8 MB/s, and the
    mount also throttles each stream to ~14 MB/s, so shards are staged to /tmp in
    parallel first;
  * split the model across the two T4s by *bytes*, not layer count -- embed_tokens
    alone is 1.8 GB, and an even 21/21 split OOMs as soon as a 2048-token batch lands;
  * keep every transient in a function scope -- a surviving reference to a 9 GB model
    makes the next member OOM at dispatch time.
"""
import gc
import glob
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

# Must precede the first CUDA allocation: gemma2 batches are fragmented badly enough
# that `reserved but unallocated` hit 821 MB at the moment of the OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import dispatch_model
from safetensors.torch import load as load_bytes
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODE = "test"
VAL_N = 600
SEED = 20261020
MAX_LEN = 2048
TOK_BUDGET = 12288
BS_CAP = 48

MEMBERS = []
CALIB = {"T": {}, "kind": "arith", "w": 0.5}


# ---------------------------------------------------------------- template
class ProcessorPAB:
    """Verbatim copy of human_pref/data/processors.py:ProcessorPAB.

    The checkpoint was trained on this exact prompt.  The arena cells are formatted as
    their python list repr, so the strings have to match verbatim.
    """

    PROMPT_PREFIX = """Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie."""

    PROMPT_SUFFIX = "verdict is: [["

    LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]

    def __init__(self, tokenizer, max_length, support_system_role):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.support_system_role = support_system_role

    def build_conversation(self, prompts, responses_a, responses_b):
        head = "<|The Start of Conversation between a User and two Assistants|>"
        tail = "<|The End of Conversation between a User and two Assistants|>\n"
        parts = []
        for prompt, response_a, response_b in zip(prompts, responses_a, responses_b):
            if prompt is None:
                prompt = "null"
            if response_a is None:
                response_a = "null"
            if response_b is None:
                response_b = "null"
            parts.append(
                f"\n### User:\n{prompt}\n\n### Assistant A:\n{response_a}\n\n### Assistant B:\n{response_b}\n"
            )
        text = "".join(parts)
        input_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        truncated_text = self.tokenizer.decode(input_ids)
        return head + truncated_text + tail

    def build_input(self, data):
        conversation = self.build_conversation(
            json.loads(data["prompt"]),
            json.loads(data["response_a"]),
            json.loads(data["response_b"]),
        )
        if self.support_system_role:
            messages = [
                {"role": "system", "content": self.PROMPT_PREFIX},
                {"role": "user", "content": conversation},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{self.PROMPT_PREFIX}\n{conversation}"},
            ]
        input_text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            + self.PROMPT_SUFFIX
        )
        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        return dict(input_ids=input_ids, input_text=input_text)


# ---------------------------------------------------------------- utils
def log(*a):
    print(*a, flush=True)


def meminfo():
    try:
        d = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            d[k] = int(v.strip().split()[0]) // 1024
        return "ram %d/%d MB avail/total" % (d.get("MemAvailable", 0), d.get("MemTotal", 0))
    except Exception:  # noqa: BLE001
        return "ram ?"


def gpuinfo():
    if not torch.cuda.is_available():
        return "gpu ?"
    return " ".join("cuda%d %d/%d MB" % (i, torch.cuda.memory_allocated(i) >> 20,
                                        torch.cuda.get_device_properties(i).total_memory >> 20)
                    for i in range(torch.cuda.device_count()))


def resolve(sub):
    """Kaggle's mount layout nests inputs, so probe a few levels deep."""
    for pat in ("/kaggle/input/%s", "/kaggle/input/*/%s",
                "/kaggle/input/*/*/%s", "/kaggle/input/*/*/*/%s"):
        hits = sorted(glob.glob(pat % sub))
        if hits:
            return hits[0]
    return None


def find_comp_dir():
    """Pick the competition directory whose test.csv is largest.

    The public data ships a 3-row placeholder while the scored run gets the ~25k-row
    hidden set, and the two can land at different mount points (`/kaggle/input/<slug>`
    vs `/kaggle/input/competitions/<slug>`).  Choosing by file size is the only way to
    be sure we read the real file: picking the first existing path silently submits
    3 predictions, which is exactly how one submission got scored on 3 rows.
    """
    cands = {}
    for pat in ("/kaggle/input/llm-classification-finetuning",
                "/kaggle/input/*/llm-classification-finetuning",
                "/kaggle/input/*/*/llm-classification-finetuning",
                "/kaggle/input/*/*/*/llm-classification-finetuning"):
        for d in glob.glob(pat):
            f = os.path.join(d, "test.csv")
            if os.path.exists(f):
                cands[d] = os.path.getsize(f)
    if not cands:
        raise SystemExit("competition data not found under /kaggle/input")
    for d, sz in sorted(cands.items(), key=lambda kv: -kv[1]):
        log("  candidate %s test.csv %d bytes" % (d, sz))
    return max(cands.items(), key=lambda kv: kv[1])[0]


def shard_files(path):
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        return sorted(set(json.load(open(idx))["weight_map"].values()))
    return sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))


STAGED = {}


def stage(path):
    """Copy the shards to /tmp in parallel, then read from there.

    /kaggle/input is a FUSE mount that throttles *each stream* to ~14-45 MB/s and
    varies by node (the same 18.5 GB took 129 s once and 1370 s another time).  Eight
    parallel copies saturate the link, and /tmp is local disk, so the model load stops
    being the dominant cost of the run.  Falls back to the original path on failure.
    """
    key = os.path.basename(path.rstrip("/"))
    if key in STAGED:
        return STAGED[key]
    files = shard_files(path)
    total = sum(os.path.getsize(os.path.join(path, f)) for f in files)
    dst = "/tmp/ckpt_" + key
    os.makedirs(dst, exist_ok=True)
    if total < 1e6:
        STAGED[key] = path
        return path
    t0 = time.time()
    try:
        def one(f):
            src_p = os.path.join(path, f)
            dst_p = os.path.join(dst, f)
            if os.path.exists(dst_p) and os.path.getsize(dst_p) == os.path.getsize(src_p):
                return f
            shutil.copyfile(src_p, dst_p)
            return f

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(one, files))
        dt = max(time.time() - t0, 1e-6)
        log("    staged %.1f GB to %s in %.0fs (%.0f MB/s)"
            % (total / 1e9, dst, dt, total / dt / 1e6))
        STAGED[key] = dst
        return dst
    except Exception as e:  # noqa: BLE001
        log("    staging failed (%s), reading from the mount" % e)
        STAGED[key] = path
        return path


def read_sequential(path):
    """Read a shard with plain read() instead of safetensors' mmap.

    Even from /tmp, mmap page faults are the wrong way to load a 5 GB tensor file in a
    container with 32 GB RAM: one read() lets the kernel do large sequential I/Os.
    """
    t0 = time.time()
    with open(path, "rb") as f:
        n = os.fstat(f.fileno()).st_size
        try:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except Exception:  # noqa: BLE001
            pass
        buf = f.read()
    dt = max(time.time() - t0, 1e-6)
    if dt > 5:
        log("    read %.2f GB in %.0fs (%.0f MB/s)" % (n / 1e9, dt, n / dt / 1e6))
    return buf


# ---------------------------------------------------------------- model
class MLPHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, x):
        return self.score(x).float()


class Scorer(nn.Module):
    """base encoder + 3-way head, laid out as `model.*` / `head.*` like the original."""

    def __init__(self, path):
        super().__init__()
        t0 = time.time()
        cfg = AutoConfig.from_pretrained(path)
        self.cfg = cfg
        prev = torch.get_default_dtype()
        # Build straight in fp16: constructing in fp32 first peaks at 37 GB and the
        # container starts swapping.
        torch.set_default_dtype(torch.float16)
        try:
            try:
                base = AutoModel.from_config(cfg, dtype=torch.float16,
                                             attn_implementation="sdpa")
            except TypeError:
                base = AutoModel.from_config(cfg, torch_dtype=torch.float16,
                                             attn_implementation="sdpa")
        finally:
            torch.set_default_dtype(prev)
        self.model = base
        self.head = MLPHead(cfg.hidden_size)
        self.num_layers = cfg.num_hidden_layers
        log("  built %s %.1fs dtype=%s %s"
            % (type(base).__name__, time.time() - t0,
               next(base.parameters()).dtype, meminfo()))
        # config comes from the mount, the 18.5 GB of shards come from the staged copy
        self._load(stage(path))

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state
        # `norm` lives on cuda:1 while the attention_mask arrives on cuda:0, so the
        # gather index has to follow the hidden state's device, not the input's.
        idx = (attention_mask.sum(1) - 1).to(out.device)      # last real token
        h = out[torch.arange(out.size(0), device=out.device), idx]
        return self.head(h)

    def _mapped(self, sd):
        """Checkpoint -> (HF base state dict, head state dict).

        The reference checkpoints fuse the MLP as `gate_up_proj` (gate first half,
        up second half, `act(gate) * up`), while HF keeps `gate_proj` / `up_proj`
        apart.  Everything else ports over with the `model.` prefix stripped.

        The head lives at top level as `score.1.*` / `score.4.*` (Dropout sits at
        indices 0/2, GELU at 3), which is exactly what `MLPHead.score` builds, so
        those keys transfer verbatim -- only the module they go into differs.
        """
        inter = self.cfg.intermediate_size
        body, head = {}, {}
        for k, v in sd.items():
            if k.startswith("score."):
                head[k] = v
            elif k.startswith("model."):
                kk = k[6:]
                gate_up = ".mlp.gate_up_proj.weight"
                if kk.endswith(gate_up):
                    pre = kk[: -len(gate_up)]
                    body[pre + ".mlp.gate_proj.weight"] = v[:inter]
                    body[pre + ".mlp.up_proj.weight"] = v[inter:]
                else:
                    body[kk] = v
        return body, head

    def _load(self, path):
        files = shard_files(path)
        want = set(self.model.state_dict().keys()) | set(self.head.state_dict().keys())
        got = set()
        t0 = time.time()
        for i, f in enumerate(files):
            t1 = time.time()
            sd = load_bytes(read_sequential(os.path.join(path, f)))
            t_read = time.time() - t1
            body, head = self._mapped(sd)
            got |= set(body) | set(head)
            if head:
                self.head.load_state_dict(head, strict=True, assign=True)
            # assign=True swaps the Parameter objects instead of copying into them:
            # a 9B fp16 model + one 4.6GB shard would otherwise need 18+4.6+4.6 GB and
            # start swapping, which is what turned the previous load into 1.8 MB/s.
            self.model.load_state_dict(body, strict=False, assign=True)
            del sd, body, head
            gc.collect()
            log("    shard %d/%d read %.0fs total %.0fs %s"
                % (i + 1, len(files), t_read, time.time() - t0, meminfo()))
        miss = sorted(want - got)
        extra = sorted(got - want)
        log("    tensors %d/%d covered, missing=%d %s, unexpected=%d %s"
            % (len(want & got), len(want), len(miss), miss[:4], len(extra), extra[:4]))
        if miss:
            raise SystemExit("weight mapping incomplete: %d tensors missing" % len(miss))


def place(model):
    """Split across the two T4s by *bytes*, not by layer count.

    `embed_tokens` is 256k x hidden fp16 (1.8 GB and 1.1 GB), so an even layer split
    hands card 0 the embedding *plus* half the layers and leaves nowhere for the
    activations of a 2048-token batch -- gemma2 died at row 505/600 of a 14.56 GB card.
    Balancing on bytes keeps ~5 GB free per card, and keeps the two cards busy for the
    same amount of time (an unbalanced pipeline wastes the faster card).
    """
    layers = model.model.layers
    per = sum(p.numel() * p.element_size() for p in layers[0].parameters())
    emb = sum(p.numel() * p.element_size() for p in model.model.embed_tokens.parameters())
    tail = (sum(p.numel() * p.element_size() for p in model.model.norm.parameters())
            + sum(p.numel() * p.element_size() for p in model.head.parameters()))
    n = len(layers)
    k = int(round(((emb + per * n + tail) / 2 - emb) / per))   # layers on the embed card
    k = max(1, min(k, n - 1))
    dm = {"model.embed_tokens": "cuda:0"}
    for i in range(k):
        dm["model.layers.%d" % i] = "cuda:0"
    for i in range(k, n):
        dm["model.layers.%d" % i] = "cuda:1"
    dm["model.norm"] = "cuda:1"
    dm["head"] = "cuda:1"
    log("  split %d/%d layers (%.1f GB embed+%d, %.1f GB +%d)"
        % (k, n - k, (emb + per * k) / 2**30, k, (per * (n - k) + tail) / 2**30, n - k))
    return dispatch_model(model, device_map=dm, main_device="cuda:0")


@torch.inference_mode()
def score_ids(model, id_list, tag, log_every=400, stop_at=None):
    """Score `id_list`, returning (probs, scored_mask).

    `stop_at` is a wall-clock deadline: on expiry the loop stops and the mask reports
    what actually got scored, so a caller under a hard time cap can still emit a
    complete submission instead of dying.  Batches are packed to a token budget rather
    than a row count (rows differ 10x in length), and an OOM bisects the batch instead
    of killing the run -- one bad batch used to cost a whole 90-minute pass.
    """
    n = len(id_list)
    out = np.zeros((n, 3), dtype=np.float64)
    ok = np.zeros(n, dtype=bool)
    order = sorted(range(n), key=lambda i: len(id_list[i]))
    batch, cur, t0, done = [], 0, time.time(), 0

    def run(part):
        L = min(max(len(id_list[i]) for i in part), MAX_LEN)
        ids = torch.full((len(part), L), 0, dtype=torch.long)
        am = torch.zeros((len(part), L), dtype=torch.long)
        for r, i in enumerate(part):
            x = id_list[i][:L]
            ids[r, :len(x)] = torch.tensor(x, dtype=torch.long)
            am[r, :len(x)] = 1
        logits = model(input_ids=ids.to("cuda:0"), attention_mask=am.to("cuda:0"))
        return torch.softmax(logits.float(), -1).cpu().numpy()

    def flush():
        nonlocal cur, done, batch
        if not batch:
            return
        queue = [batch]
        while queue:
            part = queue.pop(0)
            try:
                p = run(part)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                if len(part) == 1:
                    raise
                mid = len(part) // 2
                # Halve the *global* token budget as well.  Bisecting alone retries the
                # same oversized batch on the next flush, so at 4096 tokens/row the pass
                # degenerates into one failed forward per batch and runs several times
                # slower than the budget assumed -- which is how a 7h plan becomes 20h.
                global TOK_BUDGET
                TOK_BUDGET = max(2048, TOK_BUDGET // 2)
                log("    %s OOM on batch of %d, bisecting; token budget -> %d"
                    % (tag, len(part), TOK_BUDGET))
                queue = [part[:mid], part[mid:]] + queue
                continue
            for r, i in enumerate(part):
                out[i] = p[r]
                ok[i] = True
            done += len(part)
            if done % log_every < len(part) or done == n:
                el = time.time() - t0
                log("    %-14s %5d/%d  %.2f rows/s  eta %.0f min"
                    % (tag, done, n, done / el, (n - done) / max(done / el, 1e-6) / 60))
        batch, cur = [], 0

    for i in order:
        if stop_at is not None and time.time() > stop_at:
            log("    %-14s DEADLINE hit, only %d/%d scored" % (tag, int(ok.sum()), n))
            break
        L = min(len(id_list[i]), MAX_LEN)
        if batch and (cur + L > TOK_BUDGET or len(batch) >= BS_CAP):
            flush()
        batch.append(i)
        cur += L
    flush()
    return out, ok


def build_ids(proc, rows, swapped):
    ids = []
    for r in rows:
        d = dict(r)
        if swapped:
            d["response_a"], d["response_b"] = r["response_b"], r["response_a"]
            d["winner_model_a"], d["winner_model_b"] = r["winner_model_b"], r["winner_model_a"]
        ids.append(proc.build_input(d)["input_ids"].tolist())
    return ids


def score_member(name, path, sys_role, rows, y):
    """Run one member end to end, returning {key: array}.  A function scope, not an
    inline loop body: a leftover reference to the 9 GB module makes the *next* member
    OOM inside dispatch_model before it scores a single row."""
    tok = AutoTokenizer.from_pretrained(path)
    proc = ProcessorPAB(tokenizer=tok, max_length=MAX_LEN, support_system_role=sys_role)
    t0 = time.time()
    ids_ab = build_ids(proc, rows, False)
    lens = [len(x) for x in ids_ab]
    log("  tokenized AB %.0fs, len mean %.0f p95 %d max %d"
        % (time.time() - t0, np.mean(lens), np.percentile(lens, 95), max(lens)))
    m = Scorer(path)
    m.eval()
    m = place(m)
    log("  dispatched, %s | %s" % (meminfo(), gpuinfo()))
    pn, _ = score_ids(m, ids_ab, name + "/AB", log_every=100)
    del ids_ab
    ids_ba = build_ids(proc, rows, True)
    ps, _ = score_ids(m, ids_ba, name + "/BA", log_every=100)
    del ids_ba
    ps = ps[:, [1, 0, 2]]
    res = {name + "_pn": pn, name + "_ps": ps, name + "_ptta": (pn + ps) / 2}
    if y is not None:
        log("  logloss AB %.5f | BA %.5f | TTA %.5f"
            % (nll(pn, y), nll(ps, y), nll(res[name + "_ptta"], y)))
        log("  mean max prob AB %.4f BA %.4f" % (pn.max(1).mean(), ps.max(1).mean()))
        log("  pred dist AB %s | order agreement %.3f"
            % (np.bincount(pn.argmax(1), minlength=3),
               (pn.argmax(1) == ps.argmax(1)).mean()))
    return res


# ---------------------------------------------------------------- calibration
def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, None)).mean())


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logp(p):
    return np.log(np.clip(p, 1e-12, None))


def fit_temperature(s, y, grid=None):
    grid = grid if grid is not None else np.arange(0.3, 6.005, 0.02)
    best = (1e9, 1.0)
    for t in grid:
        v = nll(softmax(s / t), y)
        if v < best[0]:
            best = (v, float(t))
    return best[1], best[0]


def apply_calib(probs, calib):
    names = [m[0] for m in MEMBERS]
    base = {}
    for n in names:
        s = (logp(probs[n + "_pn"]) + logp(probs[n + "_ps"])) / 2
        base[n] = s / calib["T"][n]
    w = calib["w"]
    if calib.get("kind", "arith") == "arith":
        return w * softmax(base[names[0]]) + (1 - w) * softmax(base[names[1]])
    return softmax(w * base[names[0]] + (1 - w) * softmax(base[names[1]]))


def write_submission(sub, store):
    p = apply_calib(store, CALIB)
    p = np.clip(p, 1e-6, None)
    p = p / p.sum(1, keepdims=True)
    out = pd.DataFrame({"id": sub["id"].values,
                        "winner_model_a": p[:, 0],
                        "winner_model_b": p[:, 1],
                        "winner_tie": p[:, 2]})
    out.to_csv("submission.csv", index=False)
    log("wrote submission.csv", out.shape)
    log(out.head())
    log("mean prob", p.mean(0).round(4), "| mean max", round(float(p.max(1).mean()), 4))

# ---------------------------------------------------------------- perf probe
# One free kernel run answers the four questions that decide whether both members can
# fit inside the 9 h cap:
#   1. which SDPA backend actually runs at 4096, and what forcing mem_efficient buys
#   2. whether a bigger batch (TOK_BUDGET) buys throughput, i.e. whether we are
#      compute-bound (bigger batch ~no gain) or launch/memory-bound (bigger batch wins)
#   3. the real gemma:llama cost ratio -- the scheduler has been using an assumed 1.37x
#   4. whether torchao int8 weight-only is available, and what it buys on a T4
import contextlib

LPROBE = 4096
VOCAB_GUESS = 100000
LLAMA_SHAPES = [(1, 512), (4, 512), (8, 512), (16, 512), (32, 512), (64, 512),
                (4, 1024), (8, 1024), (16, 1024), (32, 1024),
                (4, 2048), (8, 2048), (1, LPROBE)]
GEMMA_SHAPES = [(16, 512), (32, 512), (64, 512), (16, 1024), (8, 2048), (1, LPROBE)]


def make_ids(batch, L):
    g = torch.Generator().manual_seed(20261020)
    ids = torch.randint(0, VOCAB_GUESS, (batch, L), generator=g)
    return ids.to("cuda:0"), torch.ones_like(ids).to("cuda:0")


def _ctx(backend):
    if backend is None:
        return contextlib.nullcontext()
    from torch.nn.attention import sdpa_kernel
    return sdpa_kernel([backend])


def time_fwd(m, batch, L, backend=None, steps=2):
    """Best-of-`steps` forward at a fixed shape.  Random ids: timing does not care
    about content, and this keeps the probe independent of the tokenizer."""
    ids, am = make_ids(batch, L)
    with torch.inference_mode():
        with _ctx(backend):
            m(input_ids=ids, attention_mask=am)          # warmup + kernel selection
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(steps):
            t0 = time.time()
            with _ctx(backend):
                m(input_ids=ids, attention_mask=am)
            torch.cuda.synchronize()
            best = min(best, time.time() - t0)
    del ids, am
    return best, 1000.0 * best / (batch * L)


def probe_model(tag, m, shapes):
    log("-- %s: ms/tok by (backend, batch, L); lower is better" % tag)
    from torch.nn.attention import SDPBackend
    plan = [("default", None, shapes)]
    out = {}
    for bname, backend, shapes in plan:
        for batch, L in shapes:
            try:
                dt, spt = time_fwd(m, batch, L, backend)
                out[(bname, batch, L)] = spt
                log("  %-8s b=%d L=%4d  %6.2f s  %5.3f ms/tok  %s"
                    % (bname, batch, L, dt, spt, gpuinfo()))
            except Exception as e:                        # noqa: BLE001
                log("  %-8s b=%d L=%4d  FAILED %s: %s"
                    % (bname, batch, L, type(e).__name__, str(e)[:140]))
                torch.cuda.empty_cache()
    return out


def probe_int8(m, tag):
    """Pure probe: weights are converted in place, the model is discarded afterwards.
    If this works, both members shrink to ~16.8 GB combined and can stay resident,
    which removes one build+stage+load cycle on top of the speed change."""
    try:
        import torchao
        from torchao.quantization import int8_weight_only, quantize_
    except Exception as e:                                # noqa: BLE001
        log("  torchao unavailable (%s: %s) -> would need a vendored wheel"
            % (type(e).__name__, str(e)[:120]))
        return None
    log("  torchao %s present, int8 weight-only on %s"
        % (getattr(torchao, "__version__", "?"), tag))
    try:
        t0 = time.time()
        quantize_(m.model, int8_weight_only())
        torch.cuda.empty_cache()
        log("  quantized in %.0fs, %s" % (time.time() - t0, gpuinfo()))
        for batch, L in ((1, LPROBE), (2, LPROBE), (3, LPROBE)):
            try:
                dt, spt = time_fwd(m, batch, L)
                log("  int8     b=%d L=%4d  %6.2f s  %5.3f ms/tok  %s"
                    % (batch, L, dt, spt, gpuinfo()))
            except Exception as e:                        # noqa: BLE001
                log("  int8     b=%d L=%4d  FAILED %s: %s"
                    % (batch, L, type(e).__name__, str(e)[:140]))
                torch.cuda.empty_cache()
        return True
    except Exception as e:                                # noqa: BLE001
        import traceback
        traceback.print_exc()
        log("  int8 conversion failed: %s: %s" % (type(e).__name__, str(e)[:160]))
        return None


def load_member(name, path):
    m = Scorer(path)
    m.eval()
    m = place(m)
    log("== %s resident, %s | %s" % (name, meminfo(), gpuinfo()))
    return m


def main():
    log("cuda", torch.cuda.is_available(), torch.cuda.device_count())
    try:
        from torch.backends.cuda import (flash_sdp_enabled, mem_efficient_sdp_enabled,
                                        math_sdp_enabled)
        log("sdpa flash=%s mem_eff=%s math=%s"
            % (flash_sdp_enabled(), mem_efficient_sdp_enabled(), math_sdp_enabled()))
    except Exception as e:                                # noqa: BLE001
        log("sdpa diag unavailable:", e)
    log("torch %s | transformers %s" % (torch.__version__,
                                        __import__("transformers").__version__))
    log("input tree:", sorted(os.listdir("/kaggle/input")))

    # llama first: it is the member the submitted plan actually runs
    lm = load_member("llama3", resolve("lmsys-checkpoints-3-0805"))
    probe_model("llama3", lm, LLAMA_SHAPES)
    del lm
    gc.collect()
    torch.cuda.empty_cache()
    log("llama released | %s | %s" % (meminfo(), gpuinfo()))

    # gemma: gives the true cost ratio instead of the assumed one
    gm = load_member("gemma2", resolve("lmsys-checkpoints-0-0805"))
    probe_model("gemma2", gm, GEMMA_SHAPES)
    del gm
    gc.collect()
    torch.cuda.empty_cache()
    log("PROBE DONE | %s | %s" % (meminfo(), gpuinfo()))


if __name__ == "__main__":
    main()
