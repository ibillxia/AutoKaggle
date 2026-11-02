"""Assemble submission.zip from a verified chemistry directory.

Takes the unchanged invsqrt/dqi circuits and the four-address chemistry family,
verifies the archive layout and CRCs, and writes the zip in the competition's
required member order.

Usage: python package.py --chem DIR [--out ZIP]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import env  # noqa: E402,F401


def sha(path: Path) -> str:
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--chem', type=Path, required=True,
                    help='verified directory with angle_k_*.kmb + submission.npz')
    ap.add_argument('--src', type=Path, default=env.SCRATCH / 'submission',
                    help='directory holding invsqrt.kmb and dqi.kmb')
    ap.add_argument('--out', type=Path, default=env.SCRATCH / 'submission.zip')
    args = ap.parse_args()

    chem = args.chem
    angles = sorted(chem.glob('angle_k_*.kmb'),
                    key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    if not angles:
        raise SystemExit(f'no angle_k_*.kmb in {chem}')
    # The verifier reads angle_k_0, 1, 2, ... up to the first gap.
    for k, p in enumerate(angles):
        if int(p.stem.rsplit('_', 1)[1]) != k:
            raise SystemExit(f'non-consecutive ANGLE family at {p.name} (expected {k})')
    npz = chem / 'submission.npz'
    if not npz.is_file():
        raise SystemExit(f'missing {npz}')
    for name in ('invsqrt.kmb', 'dqi.kmb'):
        if not (args.src / name).is_file():
            raise SystemExit(f'missing {args.src / name}')

    members = ([('dfthcsos/' + p.name, p) for p in angles]
               + [('dfthcsos/submission.npz', npz)]
               + [('invsqrt.kmb', args.src / 'invsqrt.kmb'),
                  ('dqi.kmb', args.src / 'dqi.kmb')])

    tmp = args.out.with_suffix('.partial.zip')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, path in members:
            z.write(path, name)
    with zipfile.ZipFile(tmp) as z:
        names = z.namelist()
        assert names == [n for n, _ in members], 'member order changed'
        assert z.testzip() is None, 'CRC failure'
        for name, path in members:
            with z.open(name) as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() != sha(path):
                    raise SystemExit(f'hash mismatch after archiving {name}')
    tmp.replace(args.out)

    report = {
        'archive': str(args.out),
        'archive_sha256': sha(args.out),
        'size_bytes': args.out.stat().st_size,
        'members': names,
        'chemistry': {
            'circuits': len(angles),
            'per_circuit_sha256': {p.name: sha(p) for p in angles[:1]},
            'submission_npz_sha256': sha(npz),
        },
        'invsqrt_sha256': sha(args.src / 'invsqrt.kmb'),
        'dqi_sha256': sha(args.src / 'dqi.kmb'),
    }
    (args.out.parent / 'archive_integrity.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ('archive', 'archive_sha256', 'size_bytes')},
                     indent=1))
    print(f'{len(names)} members: {names[0]} .. {names[-1]}')


if __name__ == '__main__':
    main()
