"""Exact spectral (Gram) factorisation of the two-body target T.

error_2 = || sum_rc X_rc (x) X_rc - T ||_F with T the (N^2, N^2) matrix
T_{pq,st} = tei_pqst + 0.5*S_pq*delta_st + 0.5*delta_pq*S_st  (indices (p,q)
and (s,t)). Writing v_rc = X_rc.ravel(), the sum is sum_rc v_rc v_rc^T, which is
exactly a Gram matrix. So the EXACT factorisation is the eigendecomposition

    T = sum_i c_i phi_i phi_i^T ,   X_i = sqrt(c_i) * unvec(phi_i),

with error_2 = 0, C = 1, K = rank(T). That needs no fitting at all -- the only
question is whether K (hence R*C and the ANGLE table N + R*B) is small enough to
beat the shipped R*C = 1950. This measures that honestly.

Usage: python m_spec_sqrt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
import env  # noqa: E402,F401
sys.path.insert(0, str(env.DATA / 'verifiers'))

from dfthcsos import verify as V  # noqa: E402


def build_T(inp) -> np.ndarray:
    n = inp.num_orbitals
    tei = np.asarray(inp.ref['tei'], dtype=np.float64).reshape(n * n, n * n)
    S = np.asarray(inp.S, dtype=np.float64)
    T = tei.copy()
    # T_{pq,ss} += 0.5*S_pq   and   T_{pp,st} += 0.5*S_st
    idx = np.arange(n)
    tt = np.zeros((n, n, n, n))
    tt += tei.reshape(n, n, n, n)
    tt[:, :, idx, idx] += 0.5 * S[:, :, None]
    tt[idx, idx, :, :] += 0.5 * S[None, :, :]
    return tt.reshape(n * n, n * n)


def main() -> None:
    npz = env.SCRATCH / 'submission' / 'dfthcsos' / 'submission.npz'
    inp = V.load_inputs(npz)
    n = inp.num_orbitals

    T = build_T(inp)
    T = 0.5 * (T + T.T)
    print(f'N = {n}, T is {T.shape[0]} x {T.shape[1]}')
    print(f'trace(T) = {np.trace(T):.6f}   (expect 53.297001)')

    ev = np.linalg.eigvalsh(T)
    print(f'eigen range {ev.min():.6e} .. {ev.max():.6f}')
    print(f'negative mass {ev[ev < 0].sum():.6e}  '
          f'(T is a Gram matrix of the X, so it must be PSD up to fit residual)')

    a = np.sort(np.abs(ev))[::-1]
    tot = a.sum()
    print()
    print('Exact factorisation truncated to the K largest |c_i|:')
    print(f"{'K':>6} {'|c_K|':>12} {'mass':>9} {'tail':>12} {'R*C':>6} {'vs 1950':>9}")
    for K in (65, 200, 400, 800, 1200, 1600, 1950, 2400, 2926, 4000, 5776):
        tail = float(np.linalg.norm(a[K:]))
        print(f'{K:>6} {a[K - 1]:>12.5e} {100 * a[:K].sum() / tot:>8.3f}% '
              f'{tail:>12.6f} {K:>6} {K / 1950:>8.2f}x')

    print()
    print('Smallest K whose tail (== error_2, Hartree) clears the 0.020 gate:')
    cum = np.sqrt(np.cumsum(a[::-1] ** 2))  # tail norms from the small end
    # tail[K] = ||a[K:]||; walk K upward until it drops below the gate
    tails = np.concatenate([np.linalg.norm(a[K:]) for K in range(len(a) + 1)][:0])
    sq = np.cumsum((a ** 2)[::-1])[::-1]     # suffix sums of squares
    Kpass = None
    for K in range(0, len(a) + 1):
        t = np.sqrt(sq[K]) if K < len(a) else 0.0
        if t <= 0.020:
            Kpass = K
            break
    if Kpass is None:
        print('  none: the spectrum is too flat, K = rank(T) = '
              f'{int((a > 1e-12).sum())}')
    else:
        print(f'  K = {Kpass}  (R*C = {Kpass}, {Kpass / 1950:.2f}x the shipped 1950)')
        print(f'  ANGLE table N + R*B at B=76 becomes {n} + {Kpass}*{n} '
              f'= {n + Kpass * n} rows vs 76 + 65*60 = {76 + 65 * 60}')


if __name__ == '__main__':
    main()
