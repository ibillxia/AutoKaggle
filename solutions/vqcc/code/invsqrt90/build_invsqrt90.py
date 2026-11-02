"""Build the 90-qubit invsqrt circuit from the public source-exact modules.

Reproduces the four stages the notebook runs, in order:

    selected_schedule90.build_selected   -> the kozykappa 90-qubit schedule
    portable_exact_domain.optimize       -> full-domain exact gate replacement
    portable_calibration.apply_fixed     -> the certified 44-rotation calibration
    -> code/invsqrt90/invsqrt90.kmb

Each stage asserts its intermediate SHA-256 against the value the notebook
records, so a byte difference in our environment shows up immediately rather
than silently producing a different circuit.

Usage: python build_invsqrt90.py [--out FILE]
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

import kickmix as km  # noqa: E402

# The notebook's recorded intermediate digests (anazemcev source-exact V4).
RAW_SHA = '5cb5837936858ccd63dd160dd4df4031abad5dbf057ce873434e9708cd337b9e'
STRUCTURAL_SHA = '343ae65e25032fca924c4c3d719478bba9871631ead110bfaa47bc52e01daa76'
FINAL_SHA = '97f12b109eb17346991ca33e5604963a11eff1b2098586e048d1fd103de55b7c'
EXPECTED = (90, 1449, 44)          # qubits, static Toffolis, rotations


def digest_bytes(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=HERE / 'invsqrt90.kmb')
    ap.add_argument('--skip-sha', action='store_true',
                    help='do not fail when an intermediate digest differs')
    a = ap.parse_args()

    import signed_digit_invsqrt_v3 as core
    import selected_schedule90
    import portable_exact_domain as domain_compiler
    import portable_calibration as phase_calibration

    t0 = time.monotonic()
    cfg, tables, table_origin = core.load_or_build_tables()
    print(f'[1/4] tables loaded from {table_origin} ({time.monotonic()-t0:.0f}s)',
          flush=True)

    raw_inverse, schedule_report = selected_schedule90.build_selected(cfg, tables)
    print(f"[2/4] schedule: {schedule_report.get('schedule', '?')} "
          f"raw sha {schedule_report['raw_circuit_sha256'][:16]}…", flush=True)
    if not a.skip_sha and schedule_report['raw_circuit_sha256'] != RAW_SHA:
        print(f"  !! raw digest differs from the notebook's {RAW_SHA[:16]}…")

    structural, domain_report = domain_compiler.optimize_circuit(raw_inverse)
    tmp = HERE / '_structural.kmb'
    structural.to_file(str(tmp), format='kmb')
    s_sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    print(f'[3/4] exact-domain: {len(domain_report.replacements)} replacements, '
          f'sha {s_sha[:16]}…', flush=True)
    if not a.skip_sha and s_sha != STRUCTURAL_SHA:
        print(f"  !! structural digest differs from the notebook's "
              f"{STRUCTURAL_SHA[:16]}…")

    deltas = json.loads((HERE / 'invsqrt_calibration.json').read_text())
    inv = phase_calibration.apply_fixed_calibration(structural, deltas, a.out, km=km)
    st = inv.operation_statistics()
    rot = int(st.num_z_pow(min_clifford_hierarchy_level=4))
    got = (inv.num_qubits, int(st.num_toffoli_like), rot)
    print(f'[4/4] calibrated: qubits {got[0]}, static Toffoli {got[1]}, '
          f'rotations {got[2]}', flush=True)

    f_sha = hashlib.sha256(a.out.read_bytes()).hexdigest()
    print(f'      sha {f_sha[:16]}…')
    ok_sha = f_sha == FINAL_SHA
    ok_res = got == EXPECTED
    print(f"      matches notebook digest: {ok_sha};  matches (90,1449,44): {ok_res}")
    tmp.unlink(missing_ok=True)
    (HERE / 'build_report.json').write_text(json.dumps({
        'output': str(a.out), 'sha256': f_sha, 'resources': list(got),
        'matches_notebook_digest': ok_sha, 'matches_notebook_resources': ok_res,
        'raw_sha256': schedule_report['raw_circuit_sha256'],
        'structural_sha256': s_sha,
        'replacements': len(domain_report.replacements),
        'seconds': time.monotonic() - t0,
    }, indent=1))
    if not ok_res:
        raise SystemExit('resource counts differ from the notebook; not usable as-is')


if __name__ == '__main__':
    main()
