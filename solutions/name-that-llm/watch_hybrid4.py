# Watch ibillxia/ntl-hybrid4-poolv2 v1 (slug = title slug, NOT the metadata id).
# On COMPLETE submit to the competition and wait for score.
import nt_pipeline as P

REF = "ibillxia/ntl-hybrid4-poolv2"
st = P.wait_terminal(REF, hours=30)
P.log("hybrid4 status: %s" % st)
if st == "COMPLETE":
    ref, score = P.submit_kernel(REF, 1, "hybrid4: fork604b 0.604 base + poolv2_8b (w=0.070)")
    P.log("hybrid4 final: %s" % score)
P.log("=== hybrid4 watcher end ===")
