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

MODE="val"  : score both members on a labelled slice of train.csv, in both response
              orders, then fit per-member temperature + blend weight and report
              fit / held-out log loss.
MODE="test" : score the hidden test set with the fitted calibration, write submission.csv.

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

MODE = "val"
VAL_N = 900
MEMBER_L = {"gemma2": 512, "llama3": 4096}
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
    d = resolve("llm-classification-finetuning")
    if d and os.path.exists(os.path.join(d, "test.csv")):
        return d
    raise SystemExit("competition data not found under /kaggle/input")


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
                log("    %s OOM on batch of %d, bisecting" % (tag, len(part)))
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


def score_member(name, path, sys_role, rows, y, mlen=None):
    """Run one member end to end, returning {key: array}.  A function scope, not an
    inline loop body: a leftover reference to the 9 GB module makes the *next* member
    OOM inside dispatch_model before it scores a single row.

    `mlen` lets the two members run at different truncation lengths: the probe showed
    per-token cost is flat in batch size but rises steeply in L (0.74 ms/tok at 512 vs
    2.01 at 4096), so a short gemma plus a full-length llama is the only dual-member
    split that fits the 9 h cap."""
    global MAX_LEN
    if mlen is not None:
        MAX_LEN = mlen
    log("  max_len %d for %s" % (MAX_LEN, name))
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


# ---------------------------------------------------------------- main
def main():
    log("cuda", torch.cuda.is_available(),
        torch.cuda.device_count() if torch.cuda.is_available() else 0)
    log("input tree:", sorted(os.listdir("/kaggle/input")))
    for s in ("lmsys-checkpoints-0-0805", "lmsys-checkpoints-3-0805"):
        log("  resolve", s, "->", resolve(s))
    MEMBERS.extend([("gemma2", resolve("lmsys-checkpoints-0-0805"), False, MEMBER_L["gemma2"]),
                    ("llama3", resolve("lmsys-checkpoints-3-0805"), True, MEMBER_L["llama3"])])
    comp = find_comp_dir()
    log("competition dir", comp)

    if MODE == "val":
        df = pd.read_csv(os.path.join(comp, "train.csv"))
        lab = np.argmax(df[["winner_model_a", "winner_model_b", "winner_tie"]].values, 1)
        rng = np.random.RandomState(SEED)
        keep = []
        for c in range(3):
            idx = rng.permutation(np.where(lab == c)[0])[: VAL_N // 3]
            keep.extend(idx.tolist())
        keep = sorted(keep)
        sub = df.iloc[keep].reset_index(drop=True)
        rows = sub.to_dict("records")
        y = np.argmax(sub[["winner_model_a", "winner_model_b", "winner_tie"]].values, 1)
        log("val rows", len(sub), "dist", np.bincount(y))
    else:
        sub = pd.read_csv(os.path.join(comp, "test.csv"))
        log("test rows", len(sub), list(sub.columns))
        rows = sub.to_dict("records")
        y = None

    store = {}
    for name, path, sys_role, mlen in MEMBERS:
        log("== member", name, "L=%d" % mlen)
        try:
            store.update(score_member(name, path, sys_role, rows, y, mlen))
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            log("!! member %s failed, continuing" % name)
        gc.collect()
        torch.cuda.empty_cache()
        if store:
            np.savez_compressed("member_probs.npz", **store)
            log("  saved member_probs.npz (%d keys)" % len(store))

    if MODE == "test":
        write_submission(sub, store)
        return

    # ---- calibration: fit on half, report on the other half ----
    names = [m[0] for m in MEMBERS if m[0] + "_pn" in store]
    if len(names) < 2:
        log("only %d member(s) survived, skipping the blend fit" % len(names))
        return
    rng = np.random.RandomState(7)
    perm = rng.permutation(len(y))
    fit_i, ev_i = perm[: len(perm) // 2], perm[len(perm) // 2:]

    s = {n: (logp(store[n + "_pn"]) + logp(store[n + "_ps"])) / 2 for n in names}
    Ts = {}
    for n in names:
        t, v = fit_temperature(s[n][fit_i], y[fit_i])
        Ts[n] = t
        log("  T[%s] = %.2f (fit nll %.5f)" % (n, t, v))

    sub_fit = {k: v[fit_i] for k, v in store.items()}
    best = None
    for kind in ("arith", "geom"):
        for w in np.arange(0.0, 1.001, 0.05):
            cal = {"T": Ts, "kind": kind, "w": float(w)}
            v = nll(apply_calib(sub_fit, cal), y[fit_i])
            if best is None or v < best[0]:
                best = (v, kind, float(w))
    v, kind, w = best
    cal = {"T": Ts, "kind": kind, "w": w}
    log("  best blend %s w=%.2f (fit nll %.5f)" % (kind, w, v))

    sub_ev = {k: v[ev_i] for k, v in store.items()}
    log("")
    log("  === held-out (%d rows) ===" % len(ev_i))
    for n in names:
        log("  %-7s AB %.5f | TTA %.5f"
            % (n, nll(store[n + "_pn"][ev_i], y[ev_i]),
               nll(store[n + "_ptta"][ev_i], y[ev_i])))
    log("  calibrated blend : %.5f" % nll(apply_calib(sub_ev, cal), y[ev_i]))
    fixed = {"T": {n: 1.0 for n in names}, "kind": "arith", "w": 0.57}
    log("  fixed 0.57/0.43 blend (no T) : %.5f" % nll(apply_calib(sub_ev, fixed), y[ev_i]))
    log("  uniform          : %.5f" % nll(np.full((len(ev_i), 3), 1 / 3), y[ev_i]))
    log("CALIB=" + json.dumps(cal))


if __name__ == "__main__":
    main()
