# Watch ibillxia/ntl-hybrid6-poolbigswap (slug comes from the TITLE).
# hybrid6 = fork604b (LB 0.604) with its 8B member REPLACED by the poolbig checkpoint
# (stage-1 pool 34,940 -> 259,006 rows; holdout 0.4370 -> 0.5389). W/twin_w/Gaussian stack unchanged.
# Submits v1 on COMPLETE and prints the public score. Current best = 0.604.
import nt_pipeline as P

REF = "ibillxia/ntl-hybrid6-poolbigswap"
st = P.wait_terminal(REF, hours=30)
P.log("hybrid6 status: %s" % st)
if st == "COMPLETE":
    ref, score = P.submit_kernel(REF, 1, "hybrid6: fork604b with 8B member swapped to poolbig (259k pool)")
    P.log("hybrid6 final: %s" % score)
P.log("=== hybrid6 watcher end ===")
