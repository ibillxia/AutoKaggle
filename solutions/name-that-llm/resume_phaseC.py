# Resume phase C of the 2026-10-24 pipeline (aborted by the dataset_list typo).
# package -> build hybrid3 -> push -> submit -> score.
import nt_pipeline as P

ok = P.package_poolv2_lora()
P.log("package_poolv2_lora -> %s" % ok)
if not ok:
    raise SystemExit("FATAL: packaging failed")

if not P.build_hybrid3():
    raise SystemExit("FATAL: hybrid3 lint failed - not pushed")

ver = P.push_kernel("hybrid3")
if not ver:
    raise SystemExit("FATAL: hybrid3 push failed")

st = P.wait_terminal("ibillxia/ntl-hybrid3", hours=30)
if st == "COMPLETE":
    ref, score = P.submit_kernel("ibillxia/ntl-hybrid3", ver, "hybrid3: + poolv2_8b member (pool_v2 stage1)")
    P.log("hybrid3 final: %s" % score)
else:
    P.log("hybrid3 did not COMPLETE (%s)" % st)
P.log("=== phase C resume end ===")
