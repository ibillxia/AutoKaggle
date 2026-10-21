"""Environment probe for llm-classification-finetuning.

Answers four blocking questions before we spend GPU hours:
  1. which python / torch / transformers the current Kaggle image has,
  2. whether xformers + triton work (the public 0.8304 notebook needs them),
  3. how big the hidden test set actually is,
  4. whether the public checkpoint datasets mount.
"""
import os
import sys

print("python", sys.version)
print("platform", sys.platform)

import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print("  gpu%d %s %.1fGB cc%d.%d" % (i, p.name, p.total_memory / 2**30, p.major, p.minor))

try:
    import triton
    print("triton", triton.__version__)
except Exception as e:  # noqa: BLE001
    print("triton FAIL", type(e).__name__, e)

try:
    import xformers
    print("xformers", xformers.__version__)
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
    print("BlockDiagonalCausalMask OK")
except Exception as e:  # noqa: BLE001
    print("xformers FAIL", type(e).__name__, e)

for mod in ("transformers", "peft", "bitsandbytes", "accelerate", "pandas"):
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        print(mod, "FAIL", type(e).__name__, e)

print("---- /kaggle/input ----")
for d in sorted(os.listdir("/kaggle/input")):
    p = os.path.join("/kaggle/input", d)
    n = 0
    for _, _, fs in os.walk(p):
        n += len(fs)
    print("  %-45s files=%d" % (d, n))

print("---- test.csv ----")
try:
    import pandas as pd
    t = pd.read_csv("/kaggle/input/llm-classification-finetuning/test.csv")
    print("test shape", t.shape, "cols", list(t.columns))
    tr = pd.read_csv("/kaggle/input/llm-classification-finetuning/train.csv", nrows=5)
    print("train cols", list(tr.columns))
except Exception as e:  # noqa: BLE001
    print("read FAIL", type(e).__name__, e)

print("PROBE DONE")
