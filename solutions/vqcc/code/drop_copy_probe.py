"""Drop whole C (copy) slices -- the cost-bearing test for the third index.

X has shape (R, C, N, N) = (65, 30, 76, 76). Dropping a whole c-slice removes
one ensemble copy across every rho at once, which is what actually lowers C in
SpinFreeData_V1 (and therefore outer+inner state prep, T_h and Q).

Contrast with drop_rho_row_probe.py, where a single dropped ROW already broke
the gate (19.98 -> 32.98 mHa). If copies are more redundant, C is the axis.

Usage: python drop_copy_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as M  # noqa: E402


def main() -> None:
    npz = env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz'
    z = np.load(npz)
    inp = V.load_inputs(npz)
    n = 76
    di = np.arange(n)

    X = M.reconstruct_x(z['U'], inp.w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)

    def err2(xt: np.ndarray) -> float:
        rec = np.einsum('rcpq,rcst->pqst', xt, xt, optimize=True)
        d = rec - inp.ref['tei']
        d[:, :, di, di] -= 0.5 * inp.S[:, :, None]
        d[di, di, :, :] -= 0.5 * inp.S[None, :, :]
        return 1e3 * float(np.linalg.norm(d.ravel()))

    full = err2(X)
    mass_c = np.einsum('rcpq,rcpq->c', X, X)
    order = np.argsort(mass_c)  # least-mass copies first
    print(f'full: error_2 {full:.4f} mHa (gate 20), C=30')
    print(f'per-copy mass: min {mass_c.min():.4f} med {np.median(mass_c):.4f} '
          f'max {mass_c.max():.4f}')
    print()
    print(f"{'drop':>5} {'C':>3} {'error_2':>10} {'mass kept':>10} {'passes':>7}")
    for drop in (0, 1, 2, 3, 5, 8, 10, 12, 15, 18, 20):
        keep = np.ones(30, bool)
        keep[order[:drop]] = False
        xt = X[:, keep].copy()
        e = err2(xt)
        m = 100.0 * float((xt ** 2).sum() / (X ** 2).sum())
        print(f'{drop:>5} {int(keep.sum()):>3} {e:>10.4f} {m:>9.3f}% '
              f'{("yes" if e <= 20 else "NO"):>7}', flush=True)


if __name__ == '__main__':
    main()
