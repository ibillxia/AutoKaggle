"""Build the DQI decoder from the notebook's source chain.

Calls `dqi_driver.build`, which is the entry point the notebook itself uses.
It sets DQI_BORROW='affine', derives the KT12 multiplication tables from the
competition field, installs kozykappa's four scheduling gadgets and emits the
circuit with the symbolic runtime.

The notebook records the result as 1,789 qubits / 445,410 Toffolis with
R(eps) = 1.209. This driver asserts the qubit count and reports the Toffoli
count so a mismatch is visible immediately.

Usage: python build_dqi.py [--out FILE]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import env  # noqa: E402,F401
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(env.DATA / 'verifiers'))

EXPECTED_Q = 1789
EXPECTED_TOF = 445410


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=HERE / 'dqi.kmb')
    ap.add_argument('--work', type=Path, default=HERE / '_work')
    a = ap.parse_args()

    import dqi_driver

    t0 = time.monotonic()
    rep = dqi_driver.build(a.out, a.work)
    rep['seconds'] = time.monotonic() - t0
    rep['sha256'] = hashlib.sha256(a.out.read_bytes()).hexdigest()
    print(json.dumps(rep, indent=1))

    ok_q = rep['num_qubits'] == EXPECTED_Q
    ok_t = rep['toffoli_count'] == EXPECTED_TOF
    print(f"qubits    {rep['num_qubits']:5d}  expected {EXPECTED_Q}   {'OK' if ok_q else 'MISMATCH'}")
    print(f"Toffolis  {rep['toffoli_count']:5d}  expected {EXPECTED_TOF}  {'OK' if ok_t else 'MISMATCH'}")
    (HERE / 'build_report.json').write_text(json.dumps(rep, indent=1))
    if not ok_q:
        raise SystemExit('qubit count differs from the notebook; not usable as-is')


if __name__ == '__main__':
    main()
