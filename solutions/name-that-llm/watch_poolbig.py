# Watch ibillxia/ntl-train8b-poolbig (slug = title slug).
# On COMPLETE: pull cv_report.txt and log the stage2a holdout balanced accuracy
# (baseline: pool_v2 member = 0.4852). Does NOT submit anything.
import nt_pipeline as P

REF = "ibillxia/ntl-train8b-poolbig"
st = P.wait_terminal(REF, hours=14)
P.log("poolbig status: %s" % st)
if st == "COMPLETE":
    import os
    out = "poolbig-out"
    os.makedirs(out, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        api.kernels_output(REF, path=os.path.abspath(out))
        cv = os.path.join(out, "cv_report.txt")
        if os.path.exists(cv):
            txt = open(cv, encoding="utf-8").read().strip()
            P.log("poolbig CV: %s" % txt)
        else:
            P.log("poolbig output pulled but no cv_report.txt")
    except Exception as e:
        P.log("poolbig output pull FAILED: %s" % e)
P.log("=== poolbig watcher end ===")
