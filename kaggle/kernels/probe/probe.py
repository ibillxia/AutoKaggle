import sys, platform
print("py", sys.version)
print("platform", platform.platform())
try:
    import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as e: print("no torch", e)
import os
print("/kaggle/input:", os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "absent")
print("PROBE OK")
