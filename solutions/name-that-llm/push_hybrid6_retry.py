"""Push hybrid6 with weekly-GPU-quota retry, then hand off to watch_hybrid6.py.

Rule (user): push hybrid6; if "Maximum weekly GPU quota" comes back, wait 30 min and
retry, at most 6 retries (7 attempts total). On success, run watch_hybrid6.py
(waits for COMPLETE -> submits v<ver> -> prints publicScore).

Everything is appended to nt-run-logs/pipeline.log.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, "nt-run-logs")
LOG = os.path.join(LOGDIR, "pipeline.log")
PY = sys.executable
KERNEL_DIR = os.path.join(HERE, "hybrid6")
SLEEP = 30 * 60
MAX_RETRIES = 6


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))


def try_push():
    code = (
        "from kaggle.api.kaggle_api_extended import KaggleApi\n"
        "api=KaggleApi(); api.authenticate()\n"
        "r=api.kernels_push(r'%s')\n"
        "print('PUSHRESULT', r.error, r.versionNumber)\n" % KERNEL_DIR.replace("\\", "\\\\"))
    p = subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=HERE)
    return (p.stdout or "") + (p.stderr or "")


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    log("=== hybrid6 push driver start (max %d retries, %d min apart) ===" % (MAX_RETRIES, SLEEP // 60))
    ver = None
    for attempt in range(MAX_RETRIES + 1):
        out = try_push()
        log("push attempt %d/%d -> %s" % (attempt + 1, MAX_RETRIES + 1, out.strip().replace("\n", " | ")[-300:]))
        if "PUSHRESULT" in out and "None" in out.split("PUSHRESULT")[1]:
            line = out.split("PUSHRESULT")[1].strip().splitlines()[0]
            ver = int(line.split()[-1])
            log("hybrid6 PUSH OK version=%d" % ver)
            break
        if "Maximum weekly GPU quota" in out or "Maximum batch GPU session" in out:
            if attempt == MAX_RETRIES:
                log("hybrid6 PUSH GAVE UP: GPU quota still exhausted after %d retries" % MAX_RETRIES)
                return 2
            log("GPU quota blocked; sleeping %d min before retry" % (SLEEP // 60))
            time.sleep(SLEEP)
            continue
        log("hybrid6 PUSH FAILED (non-quota): %s" % out[-400:])
        return 3

    if ver is None:
        log("hybrid6 PUSH FAILED: no version returned")
        return 4

    log("handing off to watch_hybrid6.py (version %d)" % ver)
    import runpy
    runpy.run_path(os.path.join(HERE, "watch_hybrid6.py"), run_name="__main__")
    log("=== hybrid6 push driver end ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
