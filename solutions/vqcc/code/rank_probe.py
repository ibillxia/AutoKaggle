"""error_2 as a function of the per-X_rc rank cap (truncation, not a refit).

X^(rho c) = sum_b w_b u_b u_b^T is a sum of B rank-one terms, so a factorization
with B bases can only ever realize rank <= B per (rho,c). This measures how much
of the two-body target is lost when each X_rc is truncated to rank k, which is
the ceiling a B=k factorization could reach even with perfect fitting.

Usage: python rank_probe.py
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
    num_orbitals = 76
    di = np.arange(num_orbitals)

    X = M.reconstruct_x(z['U'], inp.w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    rank_r, copies_c = X.shape[0], X.shape[1]

    def err2(xt: np.ndarray) -> float:
        rec = np.einsum('rcpq,rcst->pqst', xt, xt, optimize=True)
        d = rec - inp.ref['tei']
        d[:, :, di, di] -= 0.5 * inp.S[:, :, None]
        d[di, di, :, :] -= 0.5 * inp.S[None, :, :]
        return 1e3 * float(np.linalg.norm(d.ravel()))

    print(f'X shape {X.shape}, full error_2 = {err2(X):.4f} mHa '
          f'(gate {1e3 * V.ERROR_2_THRESHOLD:.0f})')
    print()
    print(f"{'k':>4} {'error_2 mHa':>14} {'energy kept':>12} {'passes':>7}")
    for k in (60, 56, 48, 40, 32, 24, 20, 16):
        xt = np.empty_like(X)
        for r in range(rank_r):
            for c in range(copies_c):
                u, s, vt = np.linalg.svd(X[r, c])
                xt[r, c] = (u[:, :k] * s[:k]) @ vt[:k, :]
        e = err2(xt)
        kept = 100.0 * float((xt ** 2).sum() / (X ** 2).sum())
        print(f'{k:>4} {e:>14.4f} {kept:>11.2f}% {"yes" if e <= 20 else "NO":>7}',
              flush=True)


if __name__ == '__main__':
    main()
