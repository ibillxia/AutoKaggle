# Watch ibillxia/ntl-hybrid4-poolv2 v2 (slug = title slug, NOT the metadata id).
# v1 ERRORed: gaussian-stack find_checkpoint picked weightless base_config dir
# (dataset now mounts under /kaggle/input/datasets/... which sorts before /models/).
# v2 patched find_base/find_checkpoint to require weight files. On COMPLETE submit v2.
import nt_pipeline as P

REF = "ibillxia/ntl-hybrid4-poolv2"
st = P.wait_terminal(REF, hours=30)
P.log("hybrid4 status: %s" % st)
if st == "COMPLETE":
    ref, score = P.submit_kernel(REF, 2, "hybrid4 v2: fork604b 0.604 base + poolv2_8b (w=0.070), weight-aware find")
    P.log("hybrid4 final: %s" % score)
P.log("=== hybrid4 watcher end ===")
