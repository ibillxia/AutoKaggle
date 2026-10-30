# Watch ibillxia/ntl-train8b-poolbig-v9 (slug = title slug).
# On COMPLETE: pull cv_report.txt and log the stage2a holdout balanced accuracy
# (baselines: pool_v2 member = 0.4852, poolbig dirty-pool = 0.5389). Does NOT submit.
import nt_pipeline as P

REF = "ibillxia/ntl-train8b-poolbig-v9"
st = P.wait_terminal(REF, hours=16)
P.log("poolbig-v9 status: %s" % st)
if st == "COMPLETE":
    import os
    out = "poolbig-v9-out"
    os.makedirs(out, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        api.kernels_output(REF, path=os.path.abspath(out))
        cv = os.path.join(out, "cv_report.txt")
        if os.path.exists(cv):
            txt = open(cv, encoding="utf-8").read().strip()
            P.log("poolbig-v9 CV: %s" % txt)
        else:
            P.log("poolbig-v9 output pulled but no cv_report.txt")
    except Exception as e:
        P.log("poolbig-v9 output pull FAILED: %s" % e)
P.log("=== poolbig-v9 watcher end ===")
