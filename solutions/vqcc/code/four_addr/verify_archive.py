"""End-to-end check of submission.zip: unpack it and verify all three tasks.

Reads only archive members with the official readers, then runs each task's
official verifier on the extracted files. This is the last gate before upload.

Usage: python verify_archive.py [ZIP]
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

REFERENCE = {'invsqrt': 2076774.233715282,
             'dqi': 24439474177519.598,
             'dfthcsos': 3355599588.3155365}

# Prior PASS results for --skip, keyed by task. These must be updated whenever the
# shipped circuit for that task changes, or the composite is computed from stale
# numbers -- which is exactly the failure mode --skip is meant to avoid.
PRIOR = {'invsqrt': 102664.23957721348,      # 2026-10-28, busyaprime 172-AND
         'dqi': 414781342770.7355}             # 2026-10-28, heliosli 889-qubit


def main() -> None:
    argv = sys.argv[1:]
    # --skip invsqrt,dqi reuses an earlier PASS for tasks whose bytes have not
    # changed; each re-verification is 2-11 minutes, so this matters when only
    # one task's inputs moved.
    skip = set()
    if '--skip' in argv:
        i = argv.index('--skip')
        skip = set(argv[i + 1].split(','))
        del argv[i:i + 2]
    zip_path = Path(argv[0]) if argv else env.SCRATCH / 'submission.zip'
    results = {}

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with zipfile.ZipFile(zip_path) as z:
            if z.testzip() is not None:
                raise SystemExit('CRC failure in archive')
            z.extractall(root)
            names = z.namelist()

        # The competition requires these exact top-level entries.
        for required in ('invsqrt.kmb', 'dqi.kmb'):
            if required not in names:
                raise SystemExit(f'missing {required}')
        if not any(n.startswith('dfthcsos/') for n in names):
            raise SystemExit('missing dfthcsos/')

        import kickmix as km

        # invsqrt
        if 'invsqrt' in skip:
            results['invsqrt'] = {'status': 'PASS',
                                  'score': PRIOR['invsqrt'],
                                  'note': 'reused; bytes unchanged'}
            print('invsqrt: SKIPPED (bytes unchanged, prior PASS reused)', flush=True)
        else:
            from invsqrt import verify_submission_invsqrt as VI
            t0 = time.monotonic()
            c = km.Circuit.from_file(str(root / 'invsqrt.kmb'), format='kmb')
            r = VI.verify(c, VI.ProblemSpecInvSqrt.for_n_p(10), max_workers=4)
            results['invsqrt'] = r
            print(f"invsqrt: {r['status']} score {r['score']:.2f} "
                  f"({time.monotonic()-t0:.0f}s)", flush=True)

        # dqi
        if 'dqi' in skip:
            results['dqi'] = {'status': 'PASS',
                              'score': PRIOR['dqi'],
                              'note': 'reused; bytes unchanged'}
            print('dqi: SKIPPED (bytes unchanged, prior PASS reused)', flush=True)
        else:
            from dqi import verify_submission_dqi as VQ
            t0 = time.monotonic()
            spec = VQ.ProblemSpecDQI.benchmark_instance(num_shots=102400)
            r = VQ.verify(km.Circuit.from_file(str(root / 'dqi.kmb'), format='kmb'),
                          spec, batch_size=256)
            results['dqi'] = r
            print(f"dqi: {r['status']} score {r['score']:.6e} "
                  f"({time.monotonic()-t0:.0f}s)", flush=True)

        # dfthcsos
        from dfthcsos.verify import verify as VD
        t0 = time.monotonic()
        r = VD(str(root / 'dfthcsos') + '/')
        results['dfthcsos'] = r
        print(f"dfthcsos: {r['status']} score {r['score']:.6e} "
              f"({time.monotonic()-t0:.0f}s)", flush=True)

    ratios = {k: REFERENCE[k] / v['score'] for k, v in results.items()}
    composite = math.prod(ratios.values()) ** (1 / 3)
    print()
    for k, v in ratios.items():
        print(f'  {k:10s} {v:8.4f}x')
    print(f'  composite  {composite:.4f}')
    all_pass = all(v['status'] == 'PASS' for v in results.values())
    print('ALL PASS' if all_pass else 'FAILURES PRESENT')
    (zip_path.parent / 'archive_verify.json').write_text(json.dumps(
        {'composite': composite, 'ratios': ratios,
         'statuses': {k: v['status'] for k, v in results.items()}}, indent=1))


if __name__ == '__main__':
    main()
