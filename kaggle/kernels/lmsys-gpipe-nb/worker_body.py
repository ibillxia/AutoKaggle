import gc
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
