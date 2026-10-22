# Kill every stale proxy_enc / queue process. psutil.kill() silently no-ops for some
# console-attached Windows processes -- taskkill /F is the reliable path.
cd "$(dirname "$0")"
"C:/Users/deepg/.workbuddy/binaries/python/envs/default/Scripts/python.exe" - <<'EOF'
import psutil, subprocess, time

def victims():
    return [p.info["pid"] for p in psutil.process_iter(["pid", "cmdline"])
            # only the trainers; matching run_proxy_queue.sh would kill the caller itself
            if "proxy_enc.py" in " ".join(p.info["cmdline"] or [])]

for _ in range(5):
    ids = victims()
    if not ids:
        break
    for i in ids:
        subprocess.run(["taskkill", "/F", "/PID", str(i)], capture_output=True)
    time.sleep(2)
print("remaining:", victims())
EOF
