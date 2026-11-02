"""Build four-address ANGLE circuits for our submission and score them officially.

Usage:
  python make_submission.py --J 3 [--out DIR] [--verify]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import env  # noqa: E402,F401

sys.path.insert(0, str(env.DATA / 'verifiers'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import banked_qrom as Q  # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as maths  # noqa: E402
from dfthcsos.verify import load_inputs, evaluate_words, verify  # noqa: E402


def signed_gauge(words, tree, brot=10):
    """Reverse Givens with signed half-open angles: word bit (brot-1) == bit (brot-2).

    A row sign does not change u u^T, so the Hamiltonian is untouched; the drift
    certificate below is the check that the re-encoding is exact.
    """
    u = maths.reconstruct_u_vectors(words / (1 << (brot - 1)), tree)
    a = u.copy()
    ang = np.empty(words.shape, np.float64)
    for j, (p, q) in reversed(list(enumerate(tree))):
        norm = np.hypot(a[:, p], a[:, q])
        sign = np.where(np.abs(a[:, p]) < 1e-12,
                        np.where(a[:, q] > 0, -1, 1),
                        np.where(a[:, p] < 0, -1, 1))
        ang[:, j] = np.where(norm < 1e-12, 0, np.arctan2(a[:, q] * sign, a[:, p] * sign))
        a[:, p] = np.where(norm < 1e-12, 0, norm * sign)
        a[:, q] = 0
    new = np.rint(ang * ((1 << (brot - 1)) / np.pi)).astype(np.int64) % (1 << brot)
    if np.any(((new >> (brot - 1)) ^ (new >> (brot - 2))) & 1):
        raise ArithmeticError('half-open gauge failed')
    dec = maths.reconstruct_u_vectors(new / (1 << (brot - 1)), tree)
    s = np.where((u * dec).sum(axis=1) < 0, -1, 1)
    drift = float(np.max(np.abs(dec * s[:, None] - u)))
    if drift > 1e-11:
        raise ArithmeticError(f'gauge drift {drift}')
    return new, drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--J', type=int, default=3)
    ap.add_argument('--s', type=int, default=2)
    ap.add_argument('--brot', type=int, default=10)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--src', type=Path, default=None,
                    help='submission.npz to recompile (default: our scratch copy)')
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()

    src = args.src or (env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz')
    data = {k: np.array(v, copy=True)
            for k, v in np.load(src, allow_pickle=False).items()}
    words, drift = signed_gauge(np.asarray(data['angle_words'], np.int64),
                                np.asarray(data['tree_edges'], np.int64), args.brot)
    print(f'signed gauge drift {drift:.2e}', flush=True)
    data['angle_words'] = words

    deltas = Q.batch_deltas(words, args.J)
    A = words.shape[0]
    sel = (A - 1).bit_length()
    qubits = sel + args.J * args.brot + (args.brot - 1) * args.J * (2 ** args.s - 1) + 5
    print(f'A={A} J={args.J} s={args.s} circuits={len(deltas)} '
          f'projected qubits={qubits}', flush=True)

    out = args.out or (env.SCRATCH / 'gauge' / f'sub_s{args.s}_J{args.J}')
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob('angle_k_*'):
        f.unlink()

    t0 = time.monotonic()
    toff, widths = [], []
    for k, d in enumerate(deltas):
        c = Q.build_circuit(d, args.s, args.brot)
        c.to_file(str(out / f'angle_k_{k}.kmb'), format='kmb')
        st = c.operation_statistics()
        toff.append(int(st.num_toffoli_like))
        widths.append(int(c.num_qubits))
        if k == 0:
            print(f'  circuit 0: {c.num_qubits} qubits, {st.num_toffoli_like} Toffolis',
                  flush=True)
    print(f'built in {time.monotonic()-t0:.1f}s; per-circuit Toffoli '
          f'{min(toff)}..{max(toff)}, max qubits {max(widths)}', flush=True)

    np.savez(out / 'submission.npz', **data)
    inp = load_inputs(out / 'submission.npz')
    m = evaluate_words(inp, words, data['tree_edges'], args.brot)
    print(f'words: e1 {1e3*m["error_1"]:.3f}  e2 {1e3*m["error_2"]:.3f} mHa  '
          f'Lambda {m["Lambda"]:.3f}  E_gap {m["E_gap"]:.4f}  '
          f'lambda_eff {m["lambda_eff"]:.4f}', flush=True)

    if args.verify:
        t0 = time.monotonic()
        r = verify(str(out) + '/')
        print(json.dumps({k: r[k] for k in ('status', 'score', 'toffoli_count',
                                           'num_qubits', 'lambda_eff',
                                           'verified_shots')}, indent=1), flush=True)
        print('resources:', json.dumps(r['resources'], indent=1), flush=True)
        print(f'{time.monotonic()-t0:.0f}s', flush=True)
        (out / 'verify.json').write_text(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
