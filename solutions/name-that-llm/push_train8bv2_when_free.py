"""Push train8b_v2 after fork604b has claimed its GPU session, waiting for a free slot.

Order matters: fork604b (push_f604b_when_free.py) gets the next freed slot; only once
its status shows QUEUED/RUNNING do we start contesting for the remaining session with
the train8b_v2 push.
"""
import subprocess
import sys
import time

PUSH = "solutions/name-that-llm/train8b_v2"
PY = sys.executable
CMD = (f'from kaggle.api.kaggle_api_extended import KaggleApi;'
       f'api=KaggleApi(); api.authenticate();'
       f'print(api.kernels_push("{PUSH}"))')
STATUS = (f'from kaggle.api.kaggle_api_extended import KaggleApi;'
          f'api=KaggleApi(); api.authenticate();'
          f'import sys; r=api.kernels_status("ibillxia/ntl-fork604b-full-deps");'
          f'print(r.status)')

# --- phase 1: wait for fork604b to take its slot ---
for _ in range(60):                 # ~10 h at 10 min
    p = subprocess.run([PY, "-c", STATUS], capture_output=True, text=True)
    st = (p.stdout or "").strip().splitlines()[-1] if (p.stdout or "").strip() else ""
    print(f"[{time.strftime('%H:%M')}] fork604b status: {st}", flush=True)
    if "QUEUED" in st or "RUNNING" in st or "COMPLETE" in st:
        break
    if "ERROR" in st:
        # push may have failed upstream; keep waiting anyway, the push script retries
        pass
    time.sleep(600)

# --- phase 2: push train8b_v2 when a session frees ---
for _ in range(60):                 # ~10 h at 10 min
    p = subprocess.run([PY, "-c", CMD], capture_output=True, text=True)
    msg = (p.stdout or "") + (p.stderr or "")
    last = msg.strip().splitlines()[-1][:220] if msg.strip() else ""
    if '"error": null' in msg:
        print(f"[{time.strftime('%H:%M')}] pushed: {last}", flush=True)
        sys.exit(0)
    if "Maximum batch GPU session" in msg:
        print(f"[{time.strftime('%H:%M')}] slots full, waiting", flush=True)
        time.sleep(600)
        continue
    print(f"[{time.strftime('%H:%M')}] push failed, not retrying: {last}", flush=True)
    sys.exit(1)

print("gave up", flush=True)
sys.exit(1)
