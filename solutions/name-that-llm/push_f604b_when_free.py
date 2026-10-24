"""Push fork604b as soon as a batch GPU session frees up (same retry logic as push_when_free.py)."""
import subprocess
import sys
import time

PUSH = "solutions/name-that-llm/fork604b"
PY = sys.executable
CMD = (f'from kaggle.api.kaggle_api_extended import KaggleApi;'
       f'api=KaggleApi(); api.authenticate();'
       f'print(api.kernels_push("{PUSH}"))')

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
