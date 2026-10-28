# hybrid5 (ref 57743107) score-only watcher. The submission is already in; do NOT resubmit.
# Scoring can legitimately hang in PENDING for hours (7h+ seen on this comp); we just poll.
import time, sys
from kaggle.api.kaggle_api_extended import KaggleApi

REF = "57743107"


def main():
    api = KaggleApi(); api.authenticate()
    last = None
    deadline = time.time() + 24 * 3600
    while time.time() < deadline:
        try:
            subs = api.competition_submissions('name-that-llm')
            s = next((x for x in subs if str(x.ref) == REF), None)
            if s is None:
                print("submission %s not found" % REF, flush=True)
                return
            line = "%s %s score=%s" % (s.status, getattr(s, 'publicScore', None) or getattr(s, 'public_score', None), getattr(s, 'publicScoreStdDev', ''))
            line = str(s.status) + " | " + str(getattr(s, 'publicScore', None))
            if line != last:
                print(time.strftime("%F %T"), line, flush=True)
                last = line
            if "PENDING" not in str(s.status):
                print("FINAL:", line, flush=True)
                return
        except Exception as e:
            print("poll err:", str(e)[:120], flush=True)
        time.sleep(300)
    print("score timeout (24h)", flush=True)


if __name__ == "__main__":
    main()
