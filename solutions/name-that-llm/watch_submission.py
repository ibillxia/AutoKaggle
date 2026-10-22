"""Poll the newest submission of a competition until it finishes, printing status transitions."""
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = sys.argv[1] if len(sys.argv) > 1 else "name-that-llm"
REF = sys.argv[2] if len(sys.argv) > 2 else None
POLL = 180
MAX_HOURS = 6

api = KaggleApi()
api.authenticate()
deadline = time.time() + MAX_HOURS * 3600
last = None
while time.time() < deadline:
    subs = api.competition_submissions(COMP)
    target = None
    for s in subs:
        if REF is None or str(s.ref) == str(REF):
            target = s
            break
    if target is None:
        print("submission %s not found yet" % REF, flush=True)
    else:
        st = str(target.status).split(".")[-1]
        score = target.public_score
        if (st, score) != last:
            print("%s %s score=%s ref=%s" % (time.strftime("%H:%M:%S"), st, score, target.ref), flush=True)
            last = (st, score)
        if st in ("COMPLETE", "ERROR", "CANCELED"):
            break
    time.sleep(POLL)
print("watcher exit", flush=True)
