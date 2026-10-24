# Watch ibillxia/ntl-hybrid3 v1; on COMPLETE submit to the competition and wait for score.
import nt_pipeline as P

st = P.wait_terminal("ibillxia/ntl-hybrid3", hours=30)
P.log("hybrid3 status: %s" % st)
if st == "COMPLETE":
    ref, score = P.submit_kernel("ibillxia/ntl-hybrid3", 1, "hybrid3: + poolv2_8b member (pool_v2 stage1)")
    P.log("hybrid3 final: %s" % score)
P.log("=== hybrid3 watcher end ===")
