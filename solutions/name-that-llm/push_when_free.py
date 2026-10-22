"""Push hybrid2 as soon as a batch GPU session frees up.

Kaggle allows two concurrent batch GPU sessions, and both are in use (fork604b RUNNING,
hybrid QUEUED). A push while both are held returns `Maximum batch GPU session count of 2
reached`, so this waits and retries instead of losing the version number.
"""
import json
import subprocess
import sys
import time

PUSH = "solutions/name-that-llm/hybrid2"
KERNEL = "ibillxia/ntl-hybrid2"
PY = sys.executable

CHECK = f'''
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
print(api.kernels_status("{KERNEL}"))
'''

for attempt in range(80):          # ~13 h at 10 min
    r = subprocess.run([PY, "-c", CHECK], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if "Maximum batch GPU session count" in out:
        print(f"[{time.strftime('%H:%M')}] slots full, waiting", flush=True)
        time.sleep(600)
        continue
    if "404" in out or "NotFound" in out:
        print(f"[{time.strftime('%H:%M')}] not pushed yet, pushing", flush=True)
    else:
        print(f"[{time.strftime('%H:%M')}] already exists: {out.strip()[:120]}", flush=True)
        sys.exit(0)

    p = subprocess.run(
        [PY, "-c", f'from kaggle.api.kaggle_api_extended import KaggleApi;'
                   f'api=KaggleApi(); api.authenticate();'
                   f'print(api.kernels_push("{PUSH}"))'],
        capture_output=True, text=True)
    msg = (p.stdout or "") + (p.stderr or "")
    if '"error": null' in msg:
        print(f"[{time.strftime('%H:%M')}] pushed: {msg.strip().splitlines()[-1][:200]}", flush=True)
        sys.exit(0)
    print(f"[{time.strftime('%H:%M')}] push failed: {msg.strip()[-200:]}", flush=True)
    time.sleep(600)

print("gave up", flush=True)
