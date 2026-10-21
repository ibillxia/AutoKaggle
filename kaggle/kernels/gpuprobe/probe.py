import os, sys
print("py", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("dev", torch.cuda.get_device_name(0), "n", torch.cuda.device_count())
except Exception as e:
    print("torch ERR", type(e).__name__, e)
for p in ("/kaggle/input", "/kaggle/working"):
    print(p, os.listdir(p) if os.path.isdir(p) else "absent")
import importlib
for m in ("transformers","peft","safetensors","flash_linear_attention","fla"):
    try:
        mod = importlib.import_module(m); print("have", m, getattr(mod,"__version__","?"))
    except Exception as e:
        print("miss", m)
print("GPU PROBE DONE")
