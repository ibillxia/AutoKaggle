# Watch ibillxia/ntl-hybrid6-poolbigswap (slug comes from the TITLE).
# hybrid6 = fork604b (LB 0.604) with its 8B member REPLACED by the poolbig checkpoint
# (stage-1 pool 34,940 -> 259,006 rows; holdout 0.4370 -> 0.5389). W/twin_w/Gaussian stack unchanged.
# Submits v1 on COMPLETE and prints the public score. Current best = 0.604.
#
# IDEMPOTENT: this watcher process dies when its session ends, so it gets re-launched.
# Before submitting it looks for an existing submission whose description mentions
# hybrid6 and, if one exists, only re-reads that score -- never a second submission.
import nt_pipeline as P

REF = "ibillxia/ntl-hybrid6-poolbigswap"
NOTE = "hybrid6: fork604b with 8B member swapped to poolbig (259k pool)"
TAG = "hybrid6"


def existing_submission():
    """Return (ref, status, score) of a prior hybrid6 submission, else None."""
    code = (
        "from kaggle.api.kaggle_api_extended import KaggleApi\n"
        "api=KaggleApi(); api.authenticate()\n"
        "for s in api.competition_submissions('%s'):\n"
        "    d=(getattr(s,'description','') or '')\n"
        "    if '%s' in d.lower():\n"
        "        print('FOUND', s.ref, str(s.status).split('.')[-1], "
        "getattr(s,'publicScore',None) or getattr(s,'public_score',None))\n"
        "        break\n" % (P.COMP, TAG))
    rc, out, err = P.api_call(code)
    for line in (out or "").splitlines():
        if line.startswith("FOUND"):
            parts = line.split()
            return parts[1], parts[2], (parts[3] if len(parts) > 3 else None)
    return None


st = P.wait_terminal(REF, hours=40)
P.log("hybrid6 status: %s" % st)
if st == "COMPLETE":
    prior = existing_submission()
    if prior:
        ref, sstat, sscore = prior
        P.log("hybrid6 already submitted as %s (%s, score=%s) -- not resubmitting" % (ref, sstat, sscore))
        if sstat == "COMPLETE" and sscore and sscore != "None":
            score = "%s %s" % (sstat, sscore)
        else:
            score = P.wait_score(ref, hours=8)
        P.log("hybrid6 final: %s" % score)
    else:
        ref, score = P.submit_kernel(REF, 1, NOTE)
        P.log("hybrid6 final: %s" % score)
P.log("=== hybrid6 watcher end ===")
