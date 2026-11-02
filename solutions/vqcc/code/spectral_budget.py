"""Does the exact Gram factorization of T have an affordable lambda_h2?

T = sum_i c_i phi_i phi_i^T is exact (error_2 = 0) with K terms. Taking
X_i = sqrt(c_i) * unvec(phi_i) and choosing u_b = eigenvectors of X_i,
w_b = eigenvalues gives sum_b w_b u_b u_b^T = X_i with

    L1_i = ||X_i||_nuc      (sum of |eigenvalues|)
    sum_b w_b = tr(X_i)

so lambda_h2 = 0.25 * sum_i (||X_i||_nuc + |W_i + tr X_i|)^2
             >= 0.25 * sum_i ||X_i||_nuc^2          (dropping the W term)

The shipped factorization has 0.25*sum L1^2 = 139.2473 and lambda_h2 = 280.3565.
If the spectral X has a much smaller nuclear budget, the factorization is worth
building; if it is larger, this route is dead and should be recorded as such.

Usage: python spectral_budget.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402

SHIPPED_L1SQ = 139.247320   # 0.25*sum L1^2, from lambda_egap_identity.py
SHIPPED_LAM2 = 280.356471


def build_T(inp) -> np.ndarray:
    n = inp.num_orbitals
    tei = np.asarray(inp.ref['tei'], dtype=np.float64)
    S = np.asarray(inp.S, dtype=np.float64)
    idx = np.arange(n)
    tt = tei.reshape(n, n, n, n).copy()
    tt[:, :, idx, idx] += 0.5 * S[:, :, None]
    tt[idx, idx, :, :] += 0.5 * S[None, :, :]
    return tt.reshape(n * n, n * n)


def main() -> None:
    npz = env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz'
    inp = V.load_inputs(npz)
    n = inp.num_orbitals
    T = build_T(inp)
    T = 0.5 * (T + T.T)
    ev, evec = np.linalg.eigh(T)
    order = np.argsort(-np.abs(ev))
    ev, evec = ev[order], evec[:, order]

    print(f'N={n}  tr(T)={np.trace(T):.6f}')
    print(f'shipped: 0.25*sum L1^2 = {SHIPPED_L1SQ:.4f}, lambda_h2 = {SHIPPED_LAM2:.4f}')
    print()
    print(f"{'K':>5} {'error_2 mHa':>12} {'0.25*sum nuc^2':>15} {'sum|tr X|':>11} "
          f"{'nuc vs ship':>12}")
    for K in (65, 200, 300, 400, 500, 800, 1200, 1950, 2900, 5776):
        if K > len(ev):
            continue
        c = ev[:K]
        X = (np.sqrt(np.abs(c))[:, None] * evec[:, :K].T)   # (K, n^2) rows = vec(X_i)
        X = X.reshape(K, n, n)
        # symmetric already (evec of symmetrized T), but symmetrize for safety
        X = 0.5 * (X + np.swapaxes(X, -1, -2))
        nuc = np.abs(np.linalg.eigvalsh(X)).sum(axis=1)
        tr = np.trace(X, axis1=1, axis2=2)
        l1sq = float(0.25 * (nuc ** 2).sum())
        # error_2 of the K-term representation
        v = X.reshape(K, n * n)
        err = float(np.linalg.norm(T - v.T @ v))
        print(f'{K:>5} {1e3 * err:>12.4f} {l1sq:>15.4f} {np.abs(tr).sum():>11.4f} '
              f'{l1sq / SHIPPED_L1SQ:>11.3f}x', flush=True)


if __name__ == '__main__':
    main()
