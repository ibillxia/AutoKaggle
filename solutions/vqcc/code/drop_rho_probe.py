"""Entry-level probe: zero the least-significant (rho, c) entries.

X^(rho c) enters the two-body target only through sum_(rho c) X (x) X, so a
(rho, c) pair whose X_rc is tiny contributes little.

NOTE: zeroing an *entry* does not change the circuit at all -- the ANGLE table
still has R rows and the LCU sum still has R*C terms, so T and Q are untouched.
This probe therefore measures error_2 only, and cannot demonstrate a saving.
Dropping whole rho rows is the cost-bearing version: see drop_rho_row_probe.py,
which shows a single dropped row already fails the gate (19.98 -> 32.98 mHa).

Usage: python drop_rho_probe.py
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

    # Rank every (rho, c) by its own Frobenius mass and zero the smallest.
    mass = np.einsum('rcpq,rcpq->rc', X, X)
    order = np.argsort(mass.ravel())
    print(f'full error_2 = {err2(X):.4f} mHa  (gate 20)')
    print()
    print(f"{'dropped':>7} {'R':>3} {'C':>3} {'error_2 mHa':>13} {'passes':>7}")
    for drop in (0, 1, 5, 10, 20, 40, 60, 100):
        xt = X.copy()
        flat = xt.reshape(-1, n, n)
        for idx in order[:drop]:
            flat[idx] = 0.0
        e = err2(xt)
        # Zeroing entries leaves the table shape alone: R and C are unchanged.
        print(f'{drop:>7} {65:>3} {30:>3} {e:>13.4f} {("yes" if e <= 20 else "NO"):>7}',
              flush=True)


if __name__ == '__main__':
    main()
