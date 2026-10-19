"""Archive kev's own calibration report (`kev.calibrate`) for the runs worth reporting.

Fit one temperature per workload and report the group-disjoint out-of-fold arm, which is the
honest estimate of what fitting on your own labelled rows would buy.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_wincompat"))
sys.path.insert(0, str(HERE.parent / "third_party" / "kev"))

from kev.calibrate import workload_report          # noqa: E402
from kev.suite import read_json, write_json        # noqa: E402

TAGS = [
    "kev-0.8b-ft-disaster__disaster",
    "kev-4b-local__disaster",
    "kev-0.8b-local__disaster",
    "kev-4b-local__nli",
    "kev-0.8b-local__xnli",
]

if __name__ == "__main__":
    out = {}
    for tag in TAGS:
        path = HERE / "runs" / tag / "rows.json"
        if not path.exists():
            print(f"{tag}: missing rows.json, skipped")
            continue
        r = workload_report(read_json(path), samples=400)
        out[tag] = {k: r[k] for k in ("n", "shipped_temperature", "workload_temperature",
                                      "fold_temperatures", "arms", "oof_vs_shipped")}
        print(f"{tag}: shipped T={r['shipped_temperature']} workload T={r['workload_temperature']:.2f} "
              f"ece {r['arms']['shipped']['ece']} -> oof {r['arms']['workload_oof']['ece']}")
    dest = HERE.parent / "solutions" / "kaggle-kev-competitions" / "results" / "calibration.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, out)
    print(f"wrote {dest}")
