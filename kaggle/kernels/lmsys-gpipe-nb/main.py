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


# ---------------------------------------------------------------- GPipe probe
# Two worker processes (one per T4), each holding only its half of the model, run
# ScheduleGPipe micro-batches so both cards are busy at once.  The serial path gets one
# card's worth of throughput per member (its two stages run back to back), so ~2x is the
# ceiling; this probe measures llama@4096 on the 900-row val slice and checks the
# probabilities against the serial run saved by lmsys-val512-nb.
import subprocess
import sys

VAL_N = 900
SEED = 20261020
LMAX = 4096
# Rows per pipeline step.  GPipe keeps only one *micro*-batch in flight, so the memory
# peak is BATCH/N_MICRO rows at LMAX, not BATCH: 8/8 = 1 row = a 1.07 GB attention
# matrix per layer.  v6 used 16 rows with n_micro=8 (2 rows/micro) and cuda:0 died --
# the serial path never exceeded 2 rows at 4096 either, for the same reason.
BATCH = 8
N_MICRO = 8
# The stage's P2P buffers are sized at construction, so width has to be a constant per
# pipeline; buckets keep the rebuild count small while wasting <128 tokens per batch.
BUCKET = 128

WORKER_SRC = r"""import gc
import glob
import json
import os
import time
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from safetensors.torch import load as load_bytes
from transformers import AutoConfig, AutoModel

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])


def log(*a):
    print("[%d]" % RANK, *a, flush=True)


class Identity(nn.Module):
    # Pass-through stand-in for `norm` on stage 0 (stage 1 applies the real one).
    #
    # It must NOT return None: AutoModel feeds `self.norm(...)` straight into
    # last_hidden_state, so a None-returning stub turned every stage-0 output into a
    # None tensor -- the "'NoneType' object has no attribute 'shape'" that killed v6
    # and every run since, inside torch's pipeline shape checks.
    def forward(self, x=None, *a, **k):
        return x


class Inject(nn.Module):
    # stands in for embed_tokens on stage 1: the "ids" it receives are actually
    # the hidden states streamed from stage 0
    def __init__(self):
        super().__init__()
        self.hidden = None

    def forward(self, input_ids=None, *a, **k):
        return self.hidden


# uniform stage signature, so the framework can split BOTH tensors into micro-batches
# for us:
#   stage 0: forward(ids, mask)    -> (hidden, mask)   [hidden un-normalized]
#   stage 1: forward(hidden, mask) -> (logits,)        [normalized, head applied]
# The mask must travel with the activations: ScheduleGPipe.step() slices the batch into
# n_micro chunks and calls forward once per chunk, so a mask kept on the module (the v11
# attempt) still describes the whole batch and blows up the per-row gather.
class StageWrap(nn.Module):
    def __init__(self, model, first, head=None):
        super().__init__()
        self.model = model
        self.first = first
        self.head = head

    def forward(self, a, b=None):
        if self.first:
            out = self.model(input_ids=a, attention_mask=b, use_cache=False)
            return out.last_hidden_state, b
        self.model.embed_tokens.hidden = a
        B, L = b.shape
        dummy = torch.zeros(B, L, dtype=torch.long, device=b.device)
        out = self.model(input_ids=dummy, attention_mask=b, use_cache=False)
        h = out.last_hidden_state
        idx = (b.sum(1) - 1).to(h.device)
        hh = h[torch.arange(B, device=h.device), idx]
        # head keeps whatever dtype the checkpoint shipped; match it explicitly
        wd = next(self.head.parameters()).dtype
        return self.head(hh.to(wd)).float()


def shard_files(path):
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        return sorted(set(json.load(open(idx))["weight_map"].values()))
    return sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))


def read_sequential(p):
    with open(p, "rb") as f:
        return f.read()


def mapped(sd, inter):
    # checkpoint -> (HF body keys relative to the inner model, head keys)
    body, head = {}, {}
    for k, v in sd.items():
        if k.startswith("score."):
            head[k] = v
        elif k.startswith("model."):
            kk = k[6:]
            gu = ".mlp.gate_up_proj.weight"
            if kk.endswith(gu):
                pre = kk[: -len(gu)]
                body[pre + ".mlp.gate_proj.weight"] = v[:inter]
                body[pre + ".mlp.up_proj.weight"] = v[inter:]
            else:
                body[kk] = v
    return body, head


def main():
    cfg = json.load(open("/tmp/gpipe_cfg.json"))
    dev = "cuda:%d" % RANK
    torch.cuda.set_device(RANK)
    dist.init_process_group("nccl", timeout=timedelta(seconds=1800))
    log("process group up, dev", dev)

    # config.json lives only on the original mount; /tmp holds the weight shards
    ac = AutoConfig.from_pretrained(cfg["cfg_path"])
    ac.use_cache = False
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        try:
            with torch.device("meta"):
                model = AutoModel.from_config(ac, attn_implementation="sdpa")
            meta = next(model.parameters()).device.type == "meta"
            log("meta build ok:", meta)
        except Exception as e:
            log("meta build failed (%r) -> cpu build" % (e,))
            model = AutoModel.from_config(ac, attn_implementation="sdpa")
            meta = False
    finally:
        torch.set_default_dtype(prev)

    layers = model.layers
    per = sum(p.numel() * p.element_size() for p in layers[0].parameters())
    emb = sum(p.numel() * p.element_size() for p in model.embed_tokens.parameters())
    n = len(layers)
    k = int(round(((emb + per * n) / 2 - emb) / per))
    k = max(1, min(k, n - 1))
    log("layers %d, split at %d (per-layer %.2f GB)" % (n, k, per / 2**30))

    if RANK == 0:
        prefixes = ("embed_tokens.",) + tuple("layers.%d." % i for i in range(k))
    else:
        prefixes = tuple("layers.%d." % i for i in range(k, n)) + ("norm.",)
    inter = ac.intermediate_size
    head_sd = {}
    t0 = time.time()
    for f in shard_files(cfg["ckpt"]):
        sd = load_bytes(read_sequential(os.path.join(cfg["ckpt"], f)))
        body, head = mapped(sd, inter)
        keep = {kk: v for kk, v in body.items() if kk.startswith(prefixes)}
        if RANK == 1:
            head_sd.update(head)
        model.load_state_dict(keep, strict=False, assign=True)
        del sd, body, head, keep
        gc.collect()
    log("own weights loaded %.0fs" % (time.time() - t0))

    # surgery: each rank keeps only its slice of the pipeline
    if RANK == 0:
        model.norm = Identity()
        model.layers = nn.ModuleList(list(layers[:k]))
    else:
        model.embed_tokens = Inject()
        model.layers = nn.ModuleList(list(layers[k:]))
    # Rebuild rotary embeddings instead of trusting the checkpoint: their `inv_freq` is
    # a non-persistent buffer, so it is absent from the shards and stays on meta after
    # the meta build.  llama3 and gemma2 both need this (gemma2 keeps one that
    # `model.to(dev)` then refuses to copy: "Cannot copy out of meta tensor").
    nrot = 0
    for name, mod in list(model.named_modules()):
        if not type(mod).__name__.endswith("RotaryEmbedding"):
            continue
        try:
            new = type(mod)(ac).to(dev)
        except Exception:
            new = type(mod)(ac, torch.device(dev))
        if "." in name:
            parent, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, new)
        else:
            model.rotary_emb = new
        nrot += 1
    log("rotary modules rebuilt: %d" % nrot)

    head = None
    if RANK == 1:
        head = nn.Sequential(nn.Dropout(0.1),
                             nn.Linear(ac.hidden_size, ac.hidden_size // 2),
                             nn.Dropout(0.1), nn.GELU(),
                             nn.Linear(ac.hidden_size // 2, 3))
        # the checkpoint stores the head as "score.1.*" (MLPHead.score is a
        # Sequential), while a bare Sequential expects "1.*"
        head.load_state_dict({k.replace("score.", ""): v for k, v in head_sd.items()},
                             strict=True, assign=True)
        del head_sd
        # `head` is an independent module, so model.to(dev) does not touch it.
        # Leaving it on CPU is what killed v5 (shape inference hit a device mix).
        head = head.to(dev)

    # `model.to(dev)` dies on any buffer that is still on meta, and the checkpoint does
    # not always carry them all; move what has data and rebuild the rest.
    #
    # The leftovers are always *non-persistent* buffers, which by definition never
    # appear in a state dict, so they cannot be copied off meta -- they have to be
    # recomputed from the config.  Two occur here: rotary `inv_freq` (rebuilt above)
    # and gemma2's `embed_scale` (= sqrt(hidden_size)).  Anything else fails loudly
    # rather than silently running with an uninitialised value.
    model.eval()
    left = []
    for nm, buf in list(model.named_buffers()):
        if buf is None:
            continue
        if buf.device.type == "meta":
            leaf = nm.rsplit(".", 1)[-1]
            if leaf != "embed_scale":
                raise SystemExit("meta buffer %r has no known reconstruction" % nm)
            val = torch.tensor(float(ac.hidden_size) ** 0.5, dtype=buf.dtype, device=dev)
            parent = nm.rsplit(".", 1)[0] if "." in nm else ""
            setattr(model.get_submodule(parent) if parent else model, leaf, val)
            left.append(nm)
        elif buf.device != torch.device(dev):
            buf.data = buf.data.to(dev)
    for _, prm in model.named_parameters():
        if prm.device.type != "meta" and prm.device != torch.device(dev):
            prm.data = prm.data.to(dev)
    log("moved to %s, rebuilt meta buffers: %s" % (dev, left or "none"))
    gc.collect()
    log("stage ready on %s | %.1f/%.0f GB"
        % (dev, torch.cuda.memory_allocated(RANK) / 2**30,
           torch.cuda.get_device_properties(RANK).total_memory / 2**30))
    dist.barrier()

    from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

    wrap = StageWrap(model, RANK == 0, head)
    H = ac.hidden_size
    NM = cfg["n_micro"]
    BW = cfg["batch"]
    if BW % NM:
        raise SystemExit("batch %d is not divisible by n_micro %d" % (BW, NM))
    log("batch %d = %d micro x %d rows" % (BW, NM, BW // NM))

    def make_pipeline(lq):
        # Build the stage for one input width, return (stage, schedule).
        #
        # The metas describe a single *micro*-batch: ScheduleGPipe.step() receives the
        # full batch and splits it itself, then validates each chunk against these
        # shapes (v9 shipped the full-batch shape here and died on
        # "expected [8, 384] actual [1, 384]").
        #
        # `output_args` is not a nicety either: with `input_args` alone torch runs
        # init-time shape inference and dies on our two-tensor stage-0 output with
        # `'NoneType' object has no attribute 'shape'` (stage.py:1414).  Overriding
        # both sides means no dry-run forward at all -- and, since the P2P buffers are
        # then sized once per construction, every distinct width needs its own stage.
        mb = BW // NM
        msk = torch.zeros(mb, lq, dtype=torch.long, device=dev)
        hid = torch.zeros(mb, lq, H, dtype=torch.float16, device=dev)
        if RANK == 0:
            st = PipelineStage(wrap, 0, 2, torch.device(dev),
                               input_args=(msk, msk), output_args=(hid, msk))
        else:
            st = PipelineStage(wrap, 1, 2, torch.device(dev),
                               input_args=(hid, msk),
                               output_args=(torch.zeros(mb, 3, dtype=torch.float32,
                                                        device=dev),))
        try:
            sc = ScheduleGPipe(st, NM)
        except TypeError as e:
            log("ScheduleGPipe without loss_fn failed (%r), using a dummy" % (e,))
            sc = ScheduleGPipe(st, NM, loss_fn=lambda o, *a: o.sum())
        return st, sc

    Q = cfg["bucket"]
    batches = torch.load(cfg["batches"], weights_only=False)
    outs = []
    t0 = time.time()
    n_tok = 0
    sched, stage, cur = None, None, None
    for bi, b in enumerate(batches):
        ids = b["ids"].to(dev, non_blocking=True)
        mask = b["mask"].to(dev, non_blocking=True)
        lq = -(-ids.shape[1] // Q) * Q          # round the width up to the bucket
        pad = lq - ids.shape[1]
        if pad:
            ids = nn.functional.pad(ids, (0, pad))
            mask = nn.functional.pad(mask, (0, pad))
        if lq != cur:
            nb = ("first" if cur is None else
                  "re-bucket %d -> %d" % (cur, lq))
            # keep both references: the P2P state lives on the stage object
            stage, sched = make_pipeline(lq)
            cur = lq
            log("%s width %d, batch %d/%d, alloc %.1f GB"
                % (nb, lq, bi, len(batches),
                   torch.cuda.memory_allocated(RANK) / 2**30))
        with torch.no_grad():
            # step() takes the forward args *positionally*, not as one tuple, and the
            # last stage takes no args at all (its inputs arrive over P2P; the only
            # optional argument there would be a loss target).
            if RANK == 0:
                out = sched.step(ids, mask)
            else:
                try:
                    out = sched.step(return_outputs=True)
                except TypeError:
                    out = sched.step()
        if RANK == 1:
            if out is None:
                raise SystemExit("last stage returned no output (return_outputs)")
            outs.append(out.detach().float().cpu())
        n_tok += int(mask.sum())
        if bi == 0:
            log("first step done: rows %d x len %d, alloc %.1f GB, reserved %.1f GB"
                % (ids.shape[0], ids.shape[1],
                   torch.cuda.memory_allocated(RANK) / 2**30,
                   torch.cuda.memory_reserved(RANK) / 2**30))
        if (bi + 1) % 10 == 0 or bi == len(batches) - 1:
            log("batch %d/%d  %.0fs  %.3f ms/tok"
                % (bi + 1, len(batches), time.time() - t0,
                   (time.time() - t0) / n_tok * 1e3))
    sec = time.time() - t0
    if RANK == 1:
        torch.save({"logits": torch.cat(outs), "sec": sec}, cfg["out"])
        log("saved %s" % cfg["out"])
    dist.barrier()
    dist.destroy_process_group()
    log("worker exit, scoring %.0fs = %.3f ms/tok" % (sec, sec / n_tok * 1e3))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(1)
"""


def sm(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, None)).mean())


NOISE = ("frame #", "Exception raised from", "Broken pipe", "NCCL", "TCPStore",
         "Heartbeat", "should dump")


def dump_worker_log(r, keep=70):
    """Print a rank's output, minus the libtorch stack-trace spam.

    The child processes write to their own files: when a rank dies inside nccl the
    surviving rank emits one ~40-line C++ trace per second, which is enough to push
    the actual Python traceback out of Kaggle's log window.
    """
    p = "/tmp/gpipe_r%d.log" % r
    if not os.path.exists(p):
        log("---- worker %d: no log ----" % r)
        return
    raw = open(p, encoding="utf-8", errors="replace").read().splitlines()
    keep_lines = [ln for ln in raw
                  if ln.strip() and not any(n in ln for n in NOISE)]
    log("---- worker %d log: %d lines, %d informative ----" % (r, len(raw), len(keep_lines)))
    for ln in keep_lines[-keep:]:
        log("  r%d| %s" % (r, ln))


MEMBERS = [
    ("llama3", "lmsys-checkpoints-3-0805", True),
    ("gemma2", "lmsys-checkpoints-0-0805", False),
]
# probe scope: llama3 is already measured (0.494 ms/tok, nll 0.6288), so a gemma-only
# run halves the wall time while still giving the blend against the serial llama probs
ONLY = "gemma2"


def make_batches(ids, lens):
    order = np.argsort(lens, kind="stable")
    batches = []
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH].tolist()
        n_real = len(idx)
        while len(idx) % N_MICRO:
            idx.append(idx[0])                    # pad to a multiple of n_micro
        Lb = int(max(lens[i] for i in idx))
        idb = torch.zeros(len(idx), Lb, dtype=torch.long)
        mb = torch.zeros(len(idx), Lb, dtype=torch.long)
        for r, i in enumerate(idx):
            x = ids[i]
            idb[r, :len(x)] = torch.tensor(x, dtype=torch.long)
            mb[r, :len(x)] = 1
        batches.append(dict(ids=idb, mask=mb, n_real=n_real,
                            rows=[int(i) for i in idx]))
    return batches


def run_worker(ckpt, staged, batches, hidden, port=29517):
    """Spawn both pipeline ranks, wait, and return (logits, seconds)."""
    out = "/tmp/gpipe_out.pt"
    if os.path.exists(out):
        os.remove(out)
    json.dump(dict(ckpt=staged, cfg_path=ckpt, batches="/tmp/gpipe_batches.pt",
                   out=out, n_micro=N_MICRO, batch=BATCH, bucket=BUCKET,
                   hidden=hidden),
              open("/tmp/gpipe_cfg.json", "w"))
    torch.save(batches, "/tmp/gpipe_batches.pt")
    with open("/tmp/gpipe_worker.py", "w") as f:
        f.write(WORKER_SRC)

    env0 = dict(os.environ, WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT="%d" % port)
    t_start = time.time()
    logs, ps = {}, []
    for r in (0, 1):
        f = open("/tmp/gpipe_r%d.log" % r, "w")
        logs[r] = f
        ps.append(subprocess.Popen([sys.executable, "/tmp/gpipe_worker.py"],
                                   env=dict(env0, RANK=str(r)),
                                   stdout=f, stderr=subprocess.STDOUT))
    # If one rank dies the other sits in a collective until the timeout, and the
    # resulting NCCL heartbeat spam floods the kernel log until the real traceback
    # scrolls off the front.  Kill the survivor as soon as the first one exits.
    while True:
        alive = [p.poll() is None for p in ps]
        if not any(alive) or time.time() - t_start > 1500:
            break
        if sum(alive) < len(ps):
            log("rank %d exited first, killing the other"
                % [i for i, a in enumerate(alive) if not a][0])
            break
        time.sleep(1.0)
    for p in ps:
        if p.poll() is None:
            p.kill()
    rc = [p.wait() for p in ps]
    for f in logs.values():
        f.close()
    log("workers done rc=%s wall %.0fs" % (rc, time.time() - t_start))
    for r in (0, 1):
        dump_worker_log(r)
    if not os.path.exists(out):
        log("NO GPipe OUTPUT")
        return None, 0.0
    o = torch.load(out, weights_only=False)
    return o["logits"].float().numpy(), o["sec"]


def assemble(logits, batches, n):
    p = np.zeros((n, 3))
    off = 0
    for b in batches:
        # the last batch is padded up to a multiple of n_micro by repeating row 0, and
        # those extra logits are dropped here -- so the row ids have to be cut too
        seg = logits[off:off + len(b["ids"])][:b["n_real"]]
        p[b["rows"][:len(seg)]] = sm(seg)
        off += len(b["ids"])
    return p


def main():
    t0 = time.time()

    def spent():
        return time.time() - t0

    log("torch", torch.__version__, "| cuda initialized in parent:",
        torch.cuda.is_initialized())
    log("input tree:", sorted(os.listdir("/kaggle/input")))
    comp = find_comp_dir()
    df = pd.read_csv(os.path.join(comp, "train.csv"))
    lab = np.argmax(df[["winner_model_a", "winner_model_b", "winner_tie"]].values, 1)
    rng = np.random.RandomState(SEED)
    keep = []
    for c in range(3):
        keep.extend(rng.permutation(np.where(lab == c)[0])[:VAL_N // 3].tolist())
    keep = sorted(keep)
    rows = df.iloc[keep].to_dict("records")
    y = lab[keep]
    log("rows", len(rows), "dist", np.bincount(y))
    # kernel_sources land under /kaggle/input/notebooks/<owner>/<slug>/ on newer
    # images, so the flat path alone is not enough
    refs = None
    for pat in ("/kaggle/input/lmsys-val512-nb/member_probs.npz",
                "/kaggle/input/notebooks/*/*/member_probs.npz",
                "/kaggle/input/*/*/member_probs.npz"):
        hits = sorted(glob.glob(pat))
        if hits:
            refs = np.load(hits[0])
            log("refs:", hits[0], "keys", refs.files)
            break
    if refs is None:
        log("!! no val512 reference found")

    store = {}
    for mi, (tag, ds, role) in enumerate(MEMBERS):
        if ONLY and tag != ONLY:
            continue
        ckpt = resolve(ds)
        log("== %s (%s), %.0f min elapsed" % (tag, ds, spent() / 60))
        if ckpt is None:
            log("   not mounted, skipped")
            continue
        tok = AutoTokenizer.from_pretrained(ckpt)
        proc = ProcessorPAB(tokenizer=tok, max_length=LMAX, support_system_role=role)
        t1 = time.time()
        ids = [proc.build_input(dict(r))["input_ids"].tolist() for r in rows]
        lens = np.array([len(x) for x in ids])
        log("   tokenized %.0fs mean %.0f p95 %.0f max %d"
            % (time.time() - t1, lens.mean(), np.percentile(lens, 95), lens.max()))
        batches = make_batches(ids, lens)
        n_tok = int(sum(b["mask"].sum() for b in batches))
        del ids, proc, tok
        log("   batches %d, real tokens %d" % (len(batches), n_tok))
        staged = stage(ckpt)
        hidden = json.load(open(os.path.join(ckpt, "config.json")))["hidden_size"]
        logits, sec = run_worker(ckpt, staged, batches, hidden, 29517 + 11 * mi)
        if logits is None:
            continue
        p = assemble(logits, batches, len(rows))
        store[tag] = p
        # keep the probabilities: the blend weight is decided offline, where the whole
        # weight grid and a 5-fold check cost nothing
        np.savez_compressed("/kaggle/working/gpipe_%s_%d.npz" % (tag, LMAX),
                            **{tag + "_pn": p, "y": y, "keep": np.array(keep)})
        log("   saved /kaggle/working/gpipe_%s_%d.npz" % (tag, LMAX))
        log("   %s: %.0fs / %d tok = %.3f ms/tok, %.2fx vs serial | nll %.5f acc %.3f"
            % (tag, sec, n_tok, sec / n_tok * 1e3, 0.778 / (sec / len(rows)),
               nll(p, y), float((p.argmax(1) == y).mean())))
        if refs is not None and (tag + "_pn") in refs.files:
            d = np.abs(p - refs[tag + "_pn"])
            log("   vs serial %s@%d: max abs diff %.5f mean %.6f | ref nll %.5f"
                % (tag, 512 if tag == "gemma2" else 4096, d.max(), d.mean(),
                   nll(refs[tag + "_pn"], y)))
        # /tmp cannot hold both checkpoints (15.0 + 18.5 GB), so drop this one before
        # the next member stages its own copy
        if staged and staged != ckpt:
            shutil.rmtree(staged, ignore_errors=True)
            STAGED.clear()

    if "gemma2" in store and (refs is not None or "llama3" in store):
        a = store.get("llama3")
        if a is None:
            a = refs["llama3_pn"]        # serial llama@4096 on the same 900 rows
            log("   blend base: serial llama3@4096 from val512 (nll %.5f)" % nll(a, y))
        b = store["gemma2"]
        for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            log("   blend w_gemma=%.2f -> nll %.5f (acc %.3f)"
                % (w, nll(w * b + (1 - w) * a, y),
                   float(((w * b + (1 - w) * a).argmax(1) == y).mean())))
        # 5-fold check on the weight: a single number on 900 rows overfits easily
        idx = np.arange(len(y))
        rs = np.random.RandomState(7)
        folds = rs.permutation(idx) % 5
        best, bw = [], []
        for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
            tot = 0.0
            for f in range(5):
                tr, te = idx[folds != f], idx[folds == f]
                tot += nll(w * b[te] + (1 - w) * a[te], y[te]) * len(te)
            bw.append((tot / len(idx), w))
        bw.sort()
        log("   cv best w_gemma=%.2f (held nll %.5f), w=0 is %.5f"
            % (bw[0][1], bw[0][0], [v for v, w in bw if w == 0.0][0]))
    log("PROBE DONE in %.0f min" % (spent() / 60))


if __name__ == "__main__":
    main()
