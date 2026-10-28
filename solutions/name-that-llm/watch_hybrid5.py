# Watch ibillxia/ntl-hybrid5-poolbig (slug is derived from the TITLE, not the metadata id).
# hybrid5 = fork604b 0.604 base with the poolv2_8b member REPLACED by poolbig_8b
# (stage-1 pool 259k rows; stage2a holdout 0.5389 vs 0.4852 for pool_v2, 0.4370 for the old 8B).
# Submits v1 on COMPLETE and prints the public score.
import nt_pipeline as P

REF = "ibillxia/ntl-hybrid5-poolbig"
st = P.wait_terminal(REF, hours=30)
P.log("hybrid5 status: %s" % st)
if st == "COMPLETE":
    ref, score = P.submit_kernel(REF, 1, "hybrid5: fork604b 0.604 base + poolbig_8b (w=0.070, 259k pool)")
    P.log("hybrid5 final: %s" % score)
P.log("=== hybrid5 watcher end ===")
