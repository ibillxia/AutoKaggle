"""Poll a Kaggle kernel until it leaves RUNNING/QUEUED, then dump its log tail."""
import os
import sys
import tempfile
import time

from kaggle.api.kaggle_api_extended import KaggleApi

REF = sys.argv[1] if len(sys.argv) > 1 else "ibillxia/ntl-fork592"
OUT = sys.argv[2] if len(sys.argv) > 2 else "fork592-out"
POLL = 180

api = KaggleApi()
api.authenticate()
os.makedirs(OUT, exist_ok=True)
log_path = os.path.join(OUT, REF.split("/")[-1] + ".log")

while True:
    status = api.kernels_status(REF)
    st = str(status)
    for cand in ("RUNNING", "QUEUED", "COMPLETE", "ERROR", "CANCELED"):
        if cand in st:
            st = cand
            break
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), st))
    if st not in ("RUNNING", "QUEUED"):
        d = os.path.join(tempfile.gettempdir(), REF.split("/")[-1] + "_pull")
        os.makedirs(d, exist_ok=True)
        try:
            api.kernels_output(REF, path=d)
        except Exception as e:  # noqa: BLE001
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("output pull failed: %r\n" % (e,))
        break
    time.sleep(POLL)

print("done", st)
